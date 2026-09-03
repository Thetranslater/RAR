import sqlite3
from pathlib import Path

from sqlalchemy import inspect

from rar_agent.storage.database import ProjectDatabase


def test_project_database_migrates_existing_chat_schema(tmp_path: Path) -> None:
    metadata = tmp_path / ".rar"
    metadata.mkdir()
    database_path = metadata / "rar.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE chat_sessions (
            id INTEGER PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            created_at DATETIME NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            chat_session_id INTEGER NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            tool_calls_json TEXT,
            tool_call_id VARCHAR(200),
            tool_name VARCHAR(200),
            created_at DATETIME NOT NULL
        );
        CREATE TABLE workflow_runs (
            id INTEGER PRIMARY KEY,
            dataset_path TEXT NOT NULL,
            status VARCHAR(30) NOT NULL,
            stage VARCHAR(100),
            error TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        INSERT INTO chat_sessions (id, title, created_at)
        VALUES (1, 'existing chat', '2026-01-01 00:00:00');
        INSERT INTO messages (
            id, chat_session_id, role, content, created_at
        ) VALUES (1, 1, 'user', 'existing message', '2026-01-01 00:00:00');
        """
    )
    connection.commit()
    connection.close()

    database = ProjectDatabase(tmp_path)

    tables = set(inspect(database.engine).get_table_names())
    chat_columns = {
        column["name"]
        for column in inspect(database.engine).get_columns("chat_sessions")
    }
    message_columns = {
        column["name"]
        for column in inspect(database.engine).get_columns("messages")
    }
    assert {"alembic_version", "tool_calls", "model_usage"} <= tables
    assert {"updated_at", "archived_at"} <= chat_columns
    assert {"kind", "include_in_context"} <= message_columns
    assert database.get_chat(1).title == "existing chat"
    assert database.list_messages(1)[0].content == "existing message"


def test_event_messages_are_visible_but_excluded_from_model_context(
    tmp_path: Path,
) -> None:
    database = ProjectDatabase(tmp_path)
    chat = database.create_chat("events")
    database.append_message(chat, "user", "continue")
    database.append_event(chat, "workflow completed")

    visible = database.list_messages(chat)
    context = database.list_messages(chat, context_only=True)

    assert [(message.role, message.kind) for message in visible] == [
        ("user", "message"),
        ("event", "event"),
    ]
    assert [message.content for message in context] == ["continue"]
