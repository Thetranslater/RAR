"""Generic multi-round model/tool loop for ordinary Project tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

from rar_agent.agent.tools import ToolApproval, ToolDispatcher, ToolObservation
from rar_agent.models.base import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)
from rar_agent.models.scheduler import ModelScheduler
from rar_agent.storage.database import ProjectDatabase
from rar_agent.text.tokenizer import CharacterTokenizer, Tokenizer

DEFAULT_SYSTEM_PROMPT = """You are the RAR Project assistant.
Read the current Project before making assumptions. Use the supplied tools for inspection and
changes. Dataset correction, merge, validation and re-export are ordinary file-and-tool tasks.
Keep every path inside the Project. After mutations, validate relevant structured artifacts.
For questions about existing Datasets or workflow intermediates, call inspect_project first and
answer from its result; do not read source code to discover the artifact layout. Stop calling tools
as soon as you have enough evidence to answer, and never repeat a call only to reconfirm a result.
Do not claim success when a tool observation reports failure or uncertainty."""

ROUND_LIMIT_MESSAGE = (
    "已达到本次任务的工具调用轮数上限。本次任务尚未完成。"
    "你可以发送“继续”让我根据已有记录接着处理。也可以缩小需要检查的范围。"
)


class ContextBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    chat_session: int
    content: str
    tool_calls: int
    status: Literal["completed", "awaiting_approval"] = "completed"
    approvals: tuple[ToolApproval, ...] = ()
    pending: PendingAgentRun | None = None


@dataclass(frozen=True, slots=True)
class PendingAgentRun:
    chat_session: int
    assistant_content: str | None
    tool_calls: tuple[ToolCall, ...]
    approvals: tuple[ToolApproval, ...]
    enabled_tools: frozenset[str] | None
    tool_count: int


class AgentHarness:
    def __init__(
        self,
        *,
        model_client: ModelClient,
        model: str,
        dispatcher: ToolDispatcher,
        database: ProjectDatabase,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_rounds: int = 12,
        scheduler: ModelScheduler | None = None,
        tokenizer: Tokenizer | None = None,
        max_context_tokens: int = 32_768,
        reserved_output_tokens: int = 4_096,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if reserved_output_tokens < 1 or max_context_tokens <= reserved_output_tokens:
            raise ValueError("context token limits are invalid")
        self.model_client = model_client
        self.model = model
        self.dispatcher = dispatcher
        self.database = database
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.scheduler = scheduler or ModelScheduler()
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens

    async def run(
        self,
        user_message: str,
        *,
        chat_session: int | None = None,
        enabled_tools: set[str] | None = None,
        record_user_message: bool = True,
    ) -> AgentRunResult:
        if chat_session is None:
            chat_session = self.database.create_chat(user_message)
        else:
            self.database.require_active_chat(chat_session)
        if record_user_message:
            self.database.append_message(chat_session, "user", user_message)
        enabled = frozenset(enabled_tools) if enabled_tools is not None else None
        return await self._drive(
            chat_session,
            self._model_messages(chat_session, enabled),
            enabled_tools=enabled,
            tool_count=0,
        )

    async def resume(
        self,
        pending: PendingAgentRun,
        *,
        approved: bool,
    ) -> AgentRunResult:
        self.database.require_active_chat(pending.chat_session)
        model_messages = self._model_messages(
            pending.chat_session, pending.enabled_tools
        )
        self._append_assistant(
            pending.chat_session,
            model_messages,
            pending.assistant_content,
            list(pending.tool_calls),
        )
        tool_count = await self._append_tool_observations(
            pending.chat_session,
            model_messages,
            pending.tool_calls,
            approvals=pending.approvals,
            approved=approved,
            tool_count=pending.tool_count,
        )
        return await self._drive(
            pending.chat_session,
            model_messages,
            enabled_tools=pending.enabled_tools,
            tool_count=tool_count,
        )

    def _model_messages(
        self, chat_session: int, enabled_tools: frozenset[str] | None
    ) -> list[ModelMessage]:
        model_messages = [ModelMessage(role="system", content=self.system_prompt)]
        model_messages.extend(
            ModelMessage(
                role=cast(Literal["system", "user", "assistant", "tool"], message.role),
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                name=message.tool_name,
            )
            for message in self.database.list_messages(chat_session, context_only=True)
        )
        return self._trim_messages(
            model_messages, self.dispatcher.definitions(enabled_tools)
        )

    async def _drive(
        self,
        chat_session: int,
        model_messages: list[ModelMessage],
        *,
        enabled_tools: frozenset[str] | None,
        tool_count: int,
    ) -> AgentRunResult:
        definitions = self.dispatcher.definitions(enabled_tools)

        for _ in range(self.max_rounds):
            model_messages = self._trim_messages(model_messages, definitions)
            response = await self.scheduler.complete(
                self.model_client,
                ModelRequest(
                    model=self.model,
                    messages=model_messages,
                    tools=definitions,
                    temperature=0,
                ),
                workload="chat",
                workload_id=str(chat_session),
            )
            self.database.record_model_usage(
                chat_session,
                provider=self.model_client.provider,
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            content = response.content or ""
            if not response.tool_calls:
                self._append_assistant(
                    chat_session,
                    model_messages,
                    response.content,
                    [],
                )
                return AgentRunResult(chat_session, content, tool_count)

            approvals = tuple(
                approval
                for call in response.tool_calls
                if (approval := self.dispatcher.approval_for(call)) is not None
            )
            if approvals:
                pending = PendingAgentRun(
                    chat_session=chat_session,
                    assistant_content=response.content,
                    tool_calls=tuple(response.tool_calls),
                    approvals=approvals,
                    enabled_tools=enabled_tools,
                    tool_count=tool_count,
                )
                return AgentRunResult(
                    chat_session,
                    content,
                    tool_count,
                    status="awaiting_approval",
                    approvals=approvals,
                    pending=pending,
                )

            self._append_assistant(
                chat_session,
                model_messages,
                response.content,
                response.tool_calls,
            )
            tool_count = await self._append_tool_observations(
                chat_session,
                model_messages,
                tuple(response.tool_calls),
                approvals=(),
                approved=False,
                tool_count=tool_count,
            )
        self.database.append_message(chat_session, "assistant", ROUND_LIMIT_MESSAGE)
        return AgentRunResult(chat_session, ROUND_LIMIT_MESSAGE, tool_count)

    def _trim_messages(
        self,
        messages: list[ModelMessage],
        definitions: list[ToolDefinition],
    ) -> list[ModelMessage]:
        if not messages or messages[0].role != "system":
            raise ValueError("model context must start with a system message")
        budget = self.max_context_tokens - self.reserved_output_tokens
        system = messages[0]
        used = self._message_tokens(system) + self.tokenizer.count(
            json.dumps(
                [
                    definition.model_dump(mode="json")
                    for definition in definitions
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        groups: list[list[ModelMessage]] = []
        for message in messages[1:]:
            if message.role == "user" or not groups:
                groups.append([])
            groups[-1].append(message)

        selected: list[list[ModelMessage]] = []
        for group in reversed(groups):
            cost = sum(self._message_tokens(message) for message in group)
            if used + cost > budget:
                if not selected:
                    raise ContextBudgetError(
                        "The latest conversation turn exceeds the configured context budget"
                    )
                break
            selected.append(group)
            used += cost
        selected.reverse()
        return [system, *(message for group in selected for message in group)]

    def _message_tokens(self, message: ModelMessage) -> int:
        payload = json.dumps(
            message.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.tokenizer.count(payload) + 8

    def _append_assistant(
        self,
        chat_session: int,
        model_messages: list[ModelMessage],
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> None:
        self.database.append_message(
            chat_session,
            "assistant",
            content or "",
            tool_calls=tool_calls,
        )
        model_messages.append(
            ModelMessage(role="assistant", content=content, tool_calls=tool_calls)
        )

    async def _append_tool_observations(
        self,
        chat_session: int,
        model_messages: list[ModelMessage],
        calls: tuple[ToolCall, ...],
        *,
        approvals: tuple[ToolApproval, ...],
        approved: bool,
        tool_count: int,
    ) -> int:
        approval_by_call = {approval.call_id: approval for approval in approvals}
        for call in calls:
            approval = approval_by_call.get(call.call_id)
            if approval is not None and not approved:
                observation = ToolObservation(
                    False,
                    {"error": "operation denied by user"},
                    approval.effect,
                )
                status = "denied"
            else:
                observation = await self.dispatcher.dispatch(
                    call,
                    approval_granted=approval is not None and approved,
                )
                status = "succeeded" if observation.success else "failed"
            tool_count += 1
            payload = observation.model_payload()
            self.database.record_tool_call(
                chat_session,
                call,
                effect=observation.effect,
                result=payload,
                status=status,
            )
            rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.database.append_message(
                chat_session,
                "tool",
                rendered,
                tool_call_id=call.call_id,
                tool_name=call.name,
            )
            model_messages.append(
                ModelMessage(
                    role="tool",
                    content=rendered,
                    tool_call_id=call.call_id,
                    name=call.name,
                )
            )
        return tool_count
