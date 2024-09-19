from dataclasses import fields, is_dataclass  # noqa: D100
from typing import Any


def dataclass_to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Recursively converts a nested dataclass into a dictionary.

    Args:
        obj: The dataclass instance to convert.

    Returns:
        A dictionary representation of the dataclass, including nested dataclasses.

    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj
