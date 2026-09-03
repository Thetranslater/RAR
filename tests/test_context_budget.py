from pathlib import Path

import pytest

from rar_agent.agent.harness import AgentHarness, ContextBudgetError
from rar_agent.agent.tools import ToolDispatcher
from rar_agent.models.scripted import ScriptedModelClient
from rar_agent.storage.database import ProjectDatabase
from rar_agent.text.tokenizer import CharacterTokenizer


async def test_agent_context_drops_old_complete_turns(tmp_path: Path) -> None:
    database = ProjectDatabase(tmp_path)
    chat = database.create_chat("context")
    database.append_message(chat, "user", "old-user-" + "u" * 140)
    database.append_message(chat, "assistant", "old-assistant-" + "a" * 140)
    client = ScriptedModelClient(["done"])
    harness = AgentHarness(
        model_client=client,
        model="test-model",
        dispatcher=ToolDispatcher(tmp_path, []),
        database=database,
        system_prompt="system",
        tokenizer=CharacterTokenizer(),
        max_context_tokens=220,
        reserved_output_tokens=40,
    )

    await harness.run("latest")

    contents = [message.content for message in client.requests[0].messages]
    assert contents == ["system", "latest"]


async def test_agent_rejects_latest_turn_larger_than_context_budget(
    tmp_path: Path,
) -> None:
    database = ProjectDatabase(tmp_path)
    harness = AgentHarness(
        model_client=ScriptedModelClient([]),
        model="test-model",
        dispatcher=ToolDispatcher(tmp_path, []),
        database=database,
        system_prompt="system",
        tokenizer=CharacterTokenizer(),
        max_context_tokens=140,
        reserved_output_tokens=40,
    )

    with pytest.raises(ContextBudgetError):
        await harness.run("x" * 200)
