import pytest

from rar_agent.domain.models import (
    CharacterCandidate,
    CharacterProfileGenerationResult,
    CharacterProfileSelection,
)
from rar_agent.text.characters import CharacterResolver


def _candidates() -> list[CharacterCandidate]:
    return [
        CharacterCandidate(
            names=["Yukino Yukinoshita", "Yukino"],
            description="A quiet student.",
            plot_indexes=[0],
        ),
        CharacterCandidate(
            names=["Yukino", "Yukinoshita"],
            description="She is calm and observant.",
            plot_indexes=[0],
        ),
        CharacterCandidate(
            names=["Narrator"],
            description="First-person narration.",
            plot_indexes=None,
        ),
    ]


def _result(name: str, content: str = "Profile") -> CharacterProfileGenerationResult:
    return CharacterProfileGenerationResult(
        profile=CharacterProfileSelection(name=name, content=content)
    )


def test_jobs_match_the_single_character_prompt_input() -> None:
    jobs = CharacterResolver().build_jobs(_candidates(), description_budget=200)

    assert jobs[0].candidate_indexes == (0, 1)
    assert len(jobs) == 1
    assert jobs[0].request.model_dump(mode="json") == {
        "character": {
            "names": ["Yukino Yukinoshita", "Yukino", "Yukinoshita"],
            "description": "A quiet student.\nShe is calm and observant.",
        }
    }


def test_selected_formal_name_is_moved_first_without_losing_aliases() -> None:
    resolver = CharacterResolver()
    jobs = resolver.build_jobs(_candidates(), description_budget=200)

    resolved = resolver.resolve(
        jobs,
        [
            _result("Yukinoshita", "A composed and perceptive student."),
        ],
    )

    assert resolved[0].names == (
        "Yukinoshita",
        "Yukino Yukinoshita",
        "Yukino",
    )
    assert resolved[0].name == "Yukinoshita"
    assert resolved[0].aliases == ["Yukino Yukinoshita", "Yukino"]
    assert resolved[0].profile == "A composed and perceptive student."


def test_alias_overlap_merges_transitively_and_filters_vague_names() -> None:
    candidates = [
        CharacterCandidate(names=["秋吉纯", "纯"], description="第一段。"),
        CharacterCandidate(names=["小纯", "我"], description="第二段。"),
        CharacterCandidate(names=["纯", "小纯"], description="第三段。"),
        CharacterCandidate(names=["叙事者", "我"], description="旁白。"),
        CharacterCandidate(names=["秋吉同学"], description="保留完整称呼。"),
    ]

    jobs = CharacterResolver().build_jobs(candidates, description_budget=500)

    assert [job.candidate_indexes for job in jobs] == [(0, 1, 2), (4,)]
    assert jobs[0].request.character.names == ["秋吉纯", "纯", "小纯"]
    assert jobs[0].request.character.description == "第一段。\n第二段。\n第三段。"
    assert jobs[1].request.character.names == ["秋吉同学"]


def test_resolution_rejects_a_name_not_present_in_the_prompt_input() -> None:
    resolver = CharacterResolver()
    jobs = resolver.build_jobs(_candidates(), description_budget=200)

    with pytest.raises(ValueError, match=r"must come from character\.names"):
        resolver.resolve(jobs[:1], [_result("Model-created name")])
