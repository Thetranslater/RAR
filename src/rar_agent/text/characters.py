"""Deterministic preparation and validation around global character filtering."""

from __future__ import annotations

from collections import OrderedDict

from rar_agent.domain.models import (
    CharacterCandidate,
    CharacterFilterRequest,
    CharacterFilterRequestCandidate,
    CharacterFilterResult,
    CharacterGroup,
)


class CharacterResolver:
    def build_request(
        self, candidates: list[CharacterCandidate], *, description_budget: int
    ) -> CharacterFilterRequest:
        if description_budget < 0:
            raise ValueError("description_budget must not be negative")
        aggregated: OrderedDict[str, dict[str, object]] = OrderedDict()
        for index, candidate in enumerate(candidates):
            primary = candidate.names[0].strip()
            entry = aggregated.setdefault(
                primary,
                {"candidate_indexes": [], "names": [], "descriptions": []},
            )
            indexes = entry["candidate_indexes"]
            names = entry["names"]
            descriptions = entry["descriptions"]
            assert isinstance(indexes, list)
            assert isinstance(names, list)
            assert isinstance(descriptions, list)
            indexes.append(index)
            for name in candidate.names:
                if name not in names:
                    names.append(name)
            descriptions.append(candidate.description)

        per_entry_budget = description_budget // max(1, len(aggregated))
        request_candidates: list[CharacterFilterRequestCandidate] = []
        for entry in aggregated.values():
            indexes = entry["candidate_indexes"]
            names = entry["names"]
            descriptions = entry["descriptions"]
            assert isinstance(indexes, list)
            assert isinstance(names, list)
            assert isinstance(descriptions, list)
            combined = "\n".join(str(value) for value in descriptions)
            request_candidates.append(
                CharacterFilterRequestCandidate(
                    candidate_indexes=indexes,
                    names=names,
                    description=combined[:per_entry_budget],
                )
            )
        return CharacterFilterRequest(characters=request_candidates)

    def resolve(
        self,
        candidates: list[CharacterCandidate],
        model_result: CharacterFilterResult,
    ) -> CharacterFilterResult:
        seen_indexes: set[int] = set()
        validated: list[CharacterGroup] = []
        for group in model_result.characters:
            indexes = list(dict.fromkeys(group.candidate_indexes))
            if any(index < 0 or index >= len(candidates) for index in indexes):
                raise ValueError("character group references an unknown candidate")
            duplicate = seen_indexes.intersection(indexes)
            if duplicate:
                raise ValueError(
                    f"candidate indexes appear in multiple groups: {sorted(duplicate)}"
                )
            allowed_names = {
                name
                for index in indexes
                for name in candidates[index].names
            }
            if group.name not in allowed_names or any(
                alias not in allowed_names for alias in group.aliases
            ):
                raise ValueError("formal name and aliases must come from candidate names")
            aliases = [
                alias
                for alias in dict.fromkeys(group.aliases)
                if alias != group.name
            ]
            validated.append(
                CharacterGroup(
                    name=group.name,
                    aliases=aliases,
                    candidate_indexes=indexes,
                )
            )
            seen_indexes.update(indexes)
        return CharacterFilterResult(characters=self._merge_alias_overlap(validated))

    @staticmethod
    def _merge_alias_overlap(groups: list[CharacterGroup]) -> list[CharacterGroup]:
        merged: list[CharacterGroup] = []
        for group in groups:
            names = {group.name, *group.aliases}
            overlapping = [
                index
                for index, existing in enumerate(merged)
                if names.intersection({existing.name, *existing.aliases})
            ]
            if not overlapping:
                merged.append(group)
                continue
            first = overlapping[0]
            combined = [merged[index] for index in overlapping] + [group]
            formal_name = merged[first].name
            aliases = [
                name
                for item in combined
                for name in [item.name, *item.aliases]
                if name != formal_name
            ]
            candidate_indexes = [
                index for item in combined for index in item.candidate_indexes
            ]
            merged[first] = CharacterGroup(
                name=formal_name,
                aliases=list(dict.fromkeys(aliases)),
                candidate_indexes=list(dict.fromkeys(candidate_indexes)),
            )
            for index in reversed(overlapping[1:]):
                del merged[index]
        return merged
