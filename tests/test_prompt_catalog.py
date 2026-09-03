from pathlib import Path

from rar_agent.prompts.loader import PromptCatalog


def test_default_dialogue_prompt_matches_rlff_reference() -> None:
    project_root = Path(__file__).parents[1]
    catalog = PromptCatalog()

    assert catalog.load("dialogue_extraction").text == (
        project_root / "repo/RLFF_extraction/prompt/conversation_extraction.txt"
    ).read_text(encoding="utf-8")


def test_rlff_prompt_is_split_into_system_and_user_prefix() -> None:
    plot = PromptCatalog().load("plot_extraction").render({"k": "3"})
    dialogue = PromptCatalog().load("dialogue_extraction").render(
        {"plugin_a": ""}
    )
    profile = PromptCatalog().load("character_profile").render()

    assert "----------" not in plot.system
    assert "{k}" not in plot.system
    assert plot.user_prefix == "===输入==="
    assert "----------" not in dialogue.system
    assert "{plugin_a}" not in dialogue.system
    assert dialogue.user_prefix == "===输入==="
    assert '"character"' in profile.system
    assert '"names"' in profile.system
    assert '"description"' in profile.system
    assert '"profile"' in profile.system
    assert '"name"' in profile.system
    assert '"content"' in profile.system
    assert profile.user_prefix == "===输入==="
