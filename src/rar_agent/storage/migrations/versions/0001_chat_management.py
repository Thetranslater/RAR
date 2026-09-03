"""Add project chat management fields and UI-event messages.

Revision ID: 0001_chat_management
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from rar_agent.storage.database import Base

revision: str = "0001_chat_management"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(table: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if "chat_sessions" not in tables:
        Base.metadata.create_all(bind)
        return
    Base.metadata.create_all(bind)

    chat_columns = _columns("chat_sessions")
    if "updated_at" not in chat_columns:
        op.add_column(
            "chat_sessions",
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.execute("UPDATE chat_sessions SET updated_at = created_at")
    if "archived_at" not in chat_columns:
        op.add_column(
            "chat_sessions",
            sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        )

    message_columns = _columns("messages")
    if "kind" not in message_columns:
        op.add_column(
            "messages",
            sa.Column(
                "kind", sa.String(length=20), nullable=False, server_default="message"
            ),
        )
    if "include_in_context" not in message_columns:
        op.add_column(
            "messages",
            sa.Column(
                "include_in_context", sa.Boolean(), nullable=False, server_default=sa.true()
            ),
        )

    workflow_columns = _columns("workflow_runs")
    if "chat_session_id" not in workflow_columns:
        op.add_column(
            "workflow_runs",
            sa.Column("chat_session_id", sa.Integer(), nullable=True),
        )

    indexes = {index["name"] for index in inspect(bind).get_indexes("chat_sessions")}
    if "ix_chat_sessions_updated_at" not in indexes:
        op.create_index(
            "ix_chat_sessions_updated_at", "chat_sessions", ["updated_at"], unique=False
        )


def downgrade() -> None:
    op.drop_index("ix_chat_sessions_updated_at", table_name="chat_sessions")
    with op.batch_alter_table("workflow_runs") as batch:
        batch.drop_column("chat_session_id")
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("include_in_context")
        batch.drop_column("kind")
    with op.batch_alter_table("chat_sessions") as batch:
        batch.drop_column("archived_at")
        batch.drop_column("updated_at")
