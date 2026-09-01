"""Generic multi-round model/tool loop for ordinary Project tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from rar_agent.agent.tools import ToolDispatcher
from rar_agent.models.base import ModelClient, ModelMessage, ModelRequest
from rar_agent.models.scheduler import ModelScheduler
from rar_agent.storage.database import ProjectDatabase

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


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    chat_session: int
    content: str
    tool_calls: int


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
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.model_client = model_client
        self.model = model
        self.dispatcher = dispatcher
        self.database = database
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.scheduler = scheduler or ModelScheduler()

    async def run(
        self,
        user_message: str,
        *,
        chat_session: int | None = None,
        enabled_tools: set[str] | None = None,
    ) -> AgentRunResult:
        if chat_session is None:
            chat_session = self.database.create_chat(user_message)
        self.database.append_message(chat_session, "user", user_message)
        model_messages = [ModelMessage(role="system", content=self.system_prompt)]
        model_messages.extend(
            ModelMessage(
                role=message.role,  # type: ignore[arg-type]
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                name=message.tool_name,
            )
            for message in self.database.list_messages(chat_session)
        )
        tool_count = 0
        definitions = self.dispatcher.definitions(enabled_tools)

        for _ in range(self.max_rounds):
            response = await self.scheduler.complete(
                self.model_client,
                ModelRequest(
                    model=self.model,
                    messages=model_messages,
                    tools=definitions,
                    temperature=0,
                ),
            )
            self.database.record_model_usage(
                chat_session,
                provider=self.model_client.provider,
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            content = response.content or ""
            self.database.append_message(
                chat_session,
                "assistant",
                content,
                tool_calls=response.tool_calls,
            )
            model_messages.append(
                ModelMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                return AgentRunResult(chat_session, content, tool_count)

            for call in response.tool_calls:
                observation = await self.dispatcher.dispatch(call)
                tool_count += 1
                payload = observation.model_payload()
                status = "succeeded" if observation.success else "failed"
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
        self.database.append_message(chat_session, "assistant", ROUND_LIMIT_MESSAGE)
        return AgentRunResult(chat_session, ROUND_LIMIT_MESSAGE, tool_count)
