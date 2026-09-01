import pytest

from rar_agent.domain.models import (
    CharacterCandidate,
    CharacterFilterResult,
    CharacterGroup,
)
from rar_agent.text.characters import CharacterResolver


def _candidates() -> list[CharacterCandidate]:
    return [
        CharacterCandidate(
            names=["雪之下雪乃", "雪乃"],
            description="总武高中的学生。",
            plot_indexes=[0],
        ),
        CharacterCandidate(
            names=["雪之下雪乃", "雪乃"],
            description="性格冷静。",
            plot_indexes=[0],
        ),
        CharacterCandidate(names=["我"], description="第一人称叙述者。", plot_indexes=None),
    ]


def test_request_preaggregates_exact_primary_names_and_keeps_candidate_indexes() -> None:
    request = CharacterResolver().build_request(_candidates(), description_budget=40)

    assert request.characters[0].candidate_indexes == [0, 1]
    assert request.characters[0].names == ["雪之下雪乃", "雪乃"]
    assert request.characters[1].candidate_indexes == [2]


def test_resolution_rejects_names_not_present_in_the_candidates() -> None:
    with pytest.raises(ValueError, match="must come from candidate names"):
        CharacterResolver().resolve(
            _candidates(),
            CharacterFilterResult(
                characters=[
                    CharacterGroup(
                        name="模型创造的新名字", aliases=[], candidate_indexes=[0]
                    )
                ]
            ),
        )


def test_resolution_merges_groups_with_exact_alias_overlap() -> None:
    result = CharacterResolver().resolve(
        _candidates(),
        CharacterFilterResult(
            characters=[
                CharacterGroup(
                    name="雪之下雪乃", aliases=["雪乃"], candidate_indexes=[0]
                ),
                CharacterGroup(name="雪乃", aliases=[], candidate_indexes=[1]),
            ]
        ),
    )

    assert result.characters == [
        CharacterGroup(
            name="雪之下雪乃", aliases=["雪乃"], candidate_indexes=[0, 1]
        )
    ]
