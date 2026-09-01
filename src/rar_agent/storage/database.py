"""Project-local SQLite persistence for operational Agent and Workflow state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from rar_agent.models.base import ToolCall


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    call_id: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    effect: Mapped[str] = mapped_column(String(30))
    arguments_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class WorkflowRunRow(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(100))
    level: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    data_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ArtifactRefRow(Base):
    __tablename__ = "artifact_refs"

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("workflow_runs.id"), nullable=True
    )
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ModelUsageRow(Base):
    __tablename__ = "model_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=_now)


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: str
    content: str
    tool_calls: list[ToolCall]
    tool_call_id: str | None
    tool_name: str | None


@dataclass(frozen=True, slots=True)
class StoredToolCall:
    call_id: str
    name: str
    effect: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    status: str


class ProjectDatabase:
    """One operational database per Project; domain artifacts remain files."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        metadata_dir = self.project_root / ".rar"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        self.path = metadata_dir / "rar.sqlite3"
        self.engine = create_engine(f"sqlite:///{self.path.as_posix()}")
        Base.metadata.create_all(self.engine)

    def create_chat(self, title: str) -> int:
        with Session(self.engine) as session:
            row = ChatSessionRow(title=title[:200] or "New chat")
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def append_message(
        self,
        chat_session: int,
        role: str,
        content: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        encoded_calls = None
        if tool_calls:
            encoded_calls = json.dumps(
                [call.model_dump(mode="json") for call in tool_calls],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        with Session(self.engine) as session:
            session.add(
                MessageRow(
                    chat_session_id=chat_session,
                    role=role,
                    content=content,
                    tool_calls_json=encoded_calls,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
            )
            session.commit()

    def list_messages(self, chat_session: int) -> list[StoredMessage]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(MessageRow)
                .where(MessageRow.chat_session_id == chat_session)
                .order_by(MessageRow.id)
            ).all()
            return [
                StoredMessage(
                    role=row.role,
                    content=row.content,
                    tool_calls=[
                        ToolCall.model_validate(value)
                        for value in json.loads(row.tool_calls_json or "[]")
                    ],
                    tool_call_id=row.tool_call_id,
                    tool_name=row.tool_name,
                )
                for row in rows
            ]

    def record_tool_call(
        self,
        chat_session: int,
        call: ToolCall,
        *,
        effect: str,
        result: dict[str, Any],
        status: str,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ToolCallRow(
                    chat_session_id=chat_session,
                    call_id=call.call_id,
                    name=call.name,
                    effect=effect,
                    arguments_json=json.dumps(call.arguments, ensure_ascii=False),
                    result_json=json.dumps(result, ensure_ascii=False),
                    status=status,
                )
            )
            session.commit()

    def list_tool_calls(self, chat_session: int) -> list[StoredToolCall]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ToolCallRow)
                .where(ToolCallRow.chat_session_id == chat_session)
                .order_by(ToolCallRow.id)
            ).all()
            return [
                StoredToolCall(
                    call_id=row.call_id,
                    name=row.name,
                    effect=row.effect,
                    arguments=json.loads(row.arguments_json),
                    result=json.loads(row.result_json),
                    status=row.status,
                )
                for row in rows
            ]

    def record_model_usage(
        self,
        chat_session: int,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ModelUsageRow(
                    chat_session_id=chat_session,
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            session.commit()
