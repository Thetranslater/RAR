"""Shared parsing primitives for permissive model output."""


def non_negative_int(value: int | str, label: str) -> int:
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
