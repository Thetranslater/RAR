"""Project-local SQLite persistence for operational Agent and Workflow state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    create_engine,
    delete,
    event,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from rar_agent.models.base import ToolCall

MessageKind = Literal["message", "event"]
_WHITESPACE = re.compile(r"\s+")


def _now() -> datetime:
    return datetime.now(UTC)


class ChatNotFoundError(LookupError):
    pass


class ChatArchivedError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(20), default="message")
    include_in_context: Mapped[bool] = mapped_column(Boolean, default=True)
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    call_id: Mapped[str] = mapped_column(String(200))
    name: Mapped[str] = mapped_column(String(200))
    effect: Mapped[str] = mapped_column(String(30))
    arguments_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class WorkflowRunRow(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True
    )
    dataset_path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30))
    stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(100))
    level: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    data_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ArtifactRefRow(Base):
    __tablename__ = "artifact_refs"

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True
    )
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelUsageRow(Base):
    __tablename__ = "model_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@dataclass(frozen=True, slots=True)
class StoredChat:
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    role: str
    content: str
    kind: MessageKind
    include_in_context: bool
    tool_calls: list[ToolCall]
    tool_call_id: str | None
    tool_name: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredToolCall:
    call_id: str
    name: str
    effect: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    status: str


def _run_migrations(database_path: Path) -> None:
    config = Config()
    config.set_main_option(
        "script_location", str(Path(__file__).with_name("migrations"))
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "head")


def _enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _chat(row: ChatSessionRow) -> StoredChat:
    return StoredChat(row.id, row.title, row.created_at, row.updated_at, row.archived_at)


def _message(row: MessageRow) -> StoredMessage:
    return StoredMessage(
        id=row.id,
        role=row.role,
        content=row.content,
        kind=cast(MessageKind, row.kind),
        include_in_context=row.include_in_context,
        tool_calls=[
            ToolCall.model_validate(value)
            for value in json.loads(row.tool_calls_json or "[]")
        ],
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        created_at=row.created_at,
    )


class ProjectDatabase:
    """One operational database per Project; domain artifacts remain files."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        metadata_dir = self.project_root / ".rar"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        self.path = metadata_dir / "rar.sqlite3"
        _run_migrations(self.path)
        self.engine: Engine = create_engine(
            f"sqlite:///{self.path.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        event.listen(self.engine, "connect", _enable_foreign_keys)

    @staticmethod
    def normalize_title(title: str, *, limit: int = 60) -> str:
        value = _WHITESPACE.sub(" ", title).strip()
        return value[:limit] or "New chat"

    def create_chat(self, title: str) -> int:
        now = _now()
        with Session(self.engine) as session:
            row = ChatSessionRow(
                title=self.normalize_title(title), created_at=now, updated_at=now
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def get_chat(self, chat_session: int) -> StoredChat:
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, chat_session)
            if row is None:
                raise ChatNotFoundError(f"chat {chat_session} not found")
            return _chat(row)

    def require_active_chat(self, chat_session: int) -> StoredChat:
        value = self.get_chat(chat_session)
        if value.archived_at is not None:
            raise ChatArchivedError(f"chat {chat_session} is archived")
        return value

    def list_chats(self, *, archived: bool = False) -> list[StoredChat]:
        with Session(self.engine) as session:
            condition = (
                ChatSessionRow.archived_at.is_not(None)
                if archived
                else ChatSessionRow.archived_at.is_(None)
            )
            rows = session.scalars(
                select(ChatSessionRow)
                .where(condition)
                .order_by(ChatSessionRow.updated_at.desc(), ChatSessionRow.id.desc())
            ).all()
            return [_chat(row) for row in rows]

    def rename_chat(self, chat_session: int, title: str) -> StoredChat:
        normalized = self.normalize_title(title, limit=200)
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, chat_session)
            if row is None:
                raise ChatNotFoundError(f"chat {chat_session} not found")
            row.title = normalized
            row.updated_at = _now()
            session.commit()
            session.refresh(row)
            return _chat(row)

    def set_chat_archived(self, chat_session: int, archived: bool) -> StoredChat:
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, chat_session)
            if row is None:
                raise ChatNotFoundError(f"chat {chat_session} not found")
            now = _now()
            row.archived_at = now if archived else None
            row.updated_at = now
            session.commit()
            session.refresh(row)
            return _chat(row)

    def delete_chat(self, chat_session: int) -> None:
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, chat_session)
            if row is None:
                raise ChatNotFoundError(f"chat {chat_session} not found")
            session.execute(
                update(WorkflowRunRow)
                .where(WorkflowRunRow.chat_session_id == chat_session)
                .values(chat_session_id=None)
            )
            session.execute(
                delete(ModelUsageRow).where(ModelUsageRow.chat_session_id == chat_session)
            )
            session.execute(
                delete(ToolCallRow).where(ToolCallRow.chat_session_id == chat_session)
            )
            session.execute(
                delete(MessageRow).where(MessageRow.chat_session_id == chat_session)
            )
            session.delete(row)
            session.commit()

    def append_message(
        self,
        chat_session: int,
        role: str,
        content: str,
        *,
        kind: MessageKind = "message",
        include_in_context: bool = True,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> int:
        encoded_calls = None
        if tool_calls:
            encoded_calls = json.dumps(
                [call.model_dump(mode="json") for call in tool_calls],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        with Session(self.engine) as session:
            chat = session.get(ChatSessionRow, chat_session)
            if chat is None:
                raise ChatNotFoundError(f"chat {chat_session} not found")
            row = MessageRow(
                chat_session_id=chat_session,
                role=role,
                content=content,
                kind=kind,
                include_in_context=include_in_context,
                tool_calls_json=encoded_calls,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            session.add(row)
            chat.updated_at = _now()
            session.commit()
            session.refresh(row)
            return row.id

    def append_event(self, chat_session: int, content: str) -> int:
        return self.append_message(
            chat_session,
            "event",
            content,
            kind="event",
            include_in_context=False,
        )

    def list_messages(
        self,
        chat_session: int,
        *,
        before: int | None = None,
        limit: int | None = None,
        context_only: bool = False,
    ) -> list[StoredMessage]:
        self.get_chat(chat_session)
        with Session(self.engine) as session:
            query = select(MessageRow).where(MessageRow.chat_session_id == chat_session)
            if before is not None:
                query = query.where(MessageRow.id < before)
            if context_only:
                query = query.where(MessageRow.include_in_context.is_(True))
            if limit is None:
                rows = session.scalars(query.order_by(MessageRow.id)).all()
            else:
                rows = list(
                    reversed(
                        session.scalars(
                            query.order_by(MessageRow.id.desc()).limit(limit)
                        ).all()
                    )
                )
            return [_message(row) for row in rows]

    def record_tool_call(
        self,
        chat_session: int,
        call: ToolCall,
        *,
        effect: str,
        result: dict[str, Any],
        status: str,
    ) -> None:
        self.get_chat(chat_session)
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
        self.get_chat(chat_session)
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
        self.get_chat(chat_session)
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

    def create_workflow_run(
        self,
        dataset_path: str,
        *,
        chat_session: int | None = None,
        status: str = "queued",
    ) -> int:
        if chat_session is not None:
            self.get_chat(chat_session)
        now = _now()
        with Session(self.engine) as session:
            row = WorkflowRunRow(
                chat_session_id=chat_session,
                dataset_path=dataset_path,
                status=status,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def update_workflow_run(
        self,
        workflow_run: int,
        *,
        status: str,
        stage: str | None = None,
        dataset_path: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "stage": stage,
            "error": error,
            "updated_at": _now(),
        }
        if dataset_path is not None:
            values["dataset_path"] = dataset_path
        with Session(self.engine) as session:
            session.execute(
                update(WorkflowRunRow)
                .where(WorkflowRunRow.id == workflow_run)
                .values(**values)
            )
            session.commit()
