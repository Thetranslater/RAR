"""Character catalog cleanup and batch-local identity assignment."""

from __future__ import annotations

from rar_agent.workflow.manga.models import (
    CharacterAssignment,
    CharacterAssignmentPayload,
    CharacterObservation,
    NamedCharacter,
    NamedCharacterCatalog,
    NamedCharacterCatalogPayload,
)

_UNKNOWN = {"", "unknown", "null", "none"}


def clean_character_catalog(
    payload: NamedCharacterCatalogPayload,
) -> NamedCharacterCatalog:
    characters: list[NamedCharacter] = []
    positions: dict[str, int] = {}
    for raw in payload.characters:
        name = raw.name.strip()
        if name.casefold() in _UNKNOWN:
            continue
        aliases = _clean_strings(raw.aliases, excluded={name})
        description = raw.description.strip()
        if name not in positions:
            positions[name] = len(characters)
            characters.append(
                NamedCharacter(
                    name=name,
                    aliases=aliases,
                    description=description,
                )
            )
            continue
        position = positions[name]
        current = characters[position]
        merged_aliases = [*current.aliases]
        merged_aliases.extend(alias for alias in aliases if alias not in merged_aliases)
        merged_descriptions = [
            value
            for value in (current.description, description)
            if value
        ]
        characters[position] = current.model_copy(
            update={
                "aliases": merged_aliases,
                "description": "\n".join(dict.fromkeys(merged_descriptions)),
            }
        )
    return NamedCharacterCatalog(characters=characters)


def validate_character_assignments(
    payload: CharacterAssignmentPayload,
    observations: list[CharacterObservation],
    catalog: NamedCharacterCatalog,
) -> list[CharacterAssignment]:
    expected = [
        (observation.batch_index, observation.local_character_index)
        for observation in observations
    ]
    supplied: dict[tuple[int, int], str | None] = {}
    for value in payload.assignments:
        key = (
            _non_negative_int(value.batch_index, "batch_index"),
            _non_negative_int(value.local_character_index, "local_character_index"),
        )
        if key in supplied:
            raise ValueError("every local character must appear exactly once")
        supplied[key] = _canonical_name(value.name, catalog)
    if set(supplied) != set(expected) or len(supplied) != len(expected):
        raise ValueError("every local character must appear exactly once")
    return [
        CharacterAssignment(
            batch_index=batch_index,
            local_character_index=local_index,
            name=supplied[(batch_index, local_index)],
        )
        for batch_index, local_index in expected
    ]


def aggregate_character_descriptions(
    observations: list[CharacterObservation],
    assignments: list[CharacterAssignment],
) -> dict[str, str]:
    names = {
        (value.batch_index, value.local_character_index): value.name
        for value in assignments
    }
    descriptions: dict[str, list[str]] = {}
    for observation in sorted(
        observations,
        key=lambda value: (value.batch_index, value.local_character_index),
    ):
        name = names.get((observation.batch_index, observation.local_character_index))
        description = observation.description.strip()
        if name is None or not description:
            continue
        values = descriptions.setdefault(name, [])
        if description not in values:
            values.append(description)
    return {name: "\n".join(values) for name, values in descriptions.items()}


def _clean_strings(values: list[str], *, excluded: set[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if (
            normalized.casefold() in _UNKNOWN
            or normalized in excluded
            or normalized in result
        ):
            continue
        result.append(normalized)
    return result


def _non_negative_int(value: int | str, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        result = value
    else:
        normalized = value.strip()
        if not normalized.isdigit():
            raise ValueError(f"{label} must be an integer")
        result = int(normalized)
    if result < 0:
        raise ValueError(f"{label} must not be negative")
    return result


def _canonical_name(
    value: str | None,
    catalog: NamedCharacterCatalog,
) -> str | None:
    if value is None or value.strip().casefold() in _UNKNOWN:
        return None
    normalized = value.strip()
    formal = {character.name for character in catalog.characters}
    if normalized in formal:
        return normalized
    matches = {
        character.name
        for character in catalog.characters
        if normalized in character.aliases
    }
    if len(matches) != 1:
        raise ValueError(f"assignment name is not a unique catalog character: {value}")
    return next(iter(matches))
