"""Merge character observations and normalize profile-generation results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rar_agent.domain.models import (
    CharacterCandidate,
    CharacterProfileGenerationInput,
    CharacterProfileGenerationRequest,
    CharacterProfileGenerationResult,
)

UNSAFE_CHARACTER_NAMES = frozenset(
    {
        "主人公",
        "主角",
        "叙事者",
        "叙述者",
        "旁白",
        "我",
        "你",
        "他",
        "她",
        "它",
        "学生",
        "同学",
        "narrator",
        "protagonist",
        "main character",
        "i",
        "you",
        "he",
        "she",
        "it",
        "student",
    }
)


@dataclass(frozen=True, slots=True)
class CharacterProfileJob:
    candidate_indexes: tuple[int, ...]
    request: CharacterProfileGenerationRequest


@dataclass(frozen=True, slots=True)
class ResolvedCharacter:
    candidate_indexes: tuple[int, ...]
    names: tuple[str, ...]
    description: str
    profile: str

    @property
    def name(self) -> str:
        return self.names[0]

    @property
    def aliases(self) -> list[str]:
        return list(self.names[1:])


class CharacterResolver:
    """Own candidate aggregation and formal-name normalization."""

    def build_jobs(
        self,
        candidates: list[CharacterCandidate],
        *,
        description_budget: int,
    ) -> list[CharacterProfileJob]:
        if description_budget < 0:
            raise ValueError("description_budget must not be negative")

        usable_names: list[list[str]] = []
        groups: list[list[int]] = []
        for index, candidate in enumerate(candidates):
            names = self._usable_names(candidate.names)
            usable_names.append(names)
            if not names:
                continue

            name_set = set(names)
            matching = [
                position
                for position, group in enumerate(groups)
                if name_set.intersection(
                    name
                    for candidate_index in group
                    for name in usable_names[candidate_index]
                )
            ]
            if not matching:
                groups.append([index])
                continue

            target = matching[0]
            merged_indexes = [
                candidate_index
                for position in matching
                for candidate_index in groups[position]
            ]
            merged_indexes.append(index)
            groups[target] = sorted(merged_indexes)
            for position in reversed(matching[1:]):
                del groups[position]

        groups.sort(key=lambda group: group[0])
        per_character_budget = description_budget // max(1, len(groups))
        jobs: list[CharacterProfileJob] = []
        for indexes in groups:
            names = self._ordered_unique(
                name
                for candidate_index in indexes
                for name in usable_names[candidate_index]
            )
            descriptions = self._ordered_unique(
                candidates[candidate_index].description for candidate_index in indexes
            )
            jobs.append(
                CharacterProfileJob(
                    candidate_indexes=tuple(indexes),
                    request=CharacterProfileGenerationRequest(
                        character=CharacterProfileGenerationInput(
                            names=names,
                            description="\n".join(descriptions)[
                                :per_character_budget
                            ],
                        )
                    ),
                )
            )
        return jobs

    @staticmethod
    def _usable_names(names: list[str]) -> list[str]:
        return CharacterResolver._ordered_unique(
            name.strip()
            for name in names
            if name.strip().casefold() not in UNSAFE_CHARACTER_NAMES
        )

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def resolve(
        self,
        jobs: list[CharacterProfileJob],
        model_results: list[CharacterProfileGenerationResult],
    ) -> list[ResolvedCharacter]:
        if len(jobs) != len(model_results):
            raise ValueError("every character profile job requires one model result")
        resolved: list[ResolvedCharacter] = []
        for job, result in zip(jobs, model_results, strict=True):
            names = job.request.character.names
            selected_name = result.profile.name
            if selected_name not in names:
                raise ValueError("formal name must come from character.names")
            ordered_names = [
                selected_name,
                *(name for name in names if name != selected_name),
            ]
            resolved.append(
                ResolvedCharacter(
                    candidate_indexes=job.candidate_indexes,
                    names=tuple(ordered_names),
                    description=job.request.character.description,
                    profile=result.profile.content,
                )
            )
        return resolved
