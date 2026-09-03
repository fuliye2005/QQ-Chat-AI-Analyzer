"""Validation for the JSON contracts exchanged by Map and Reduce."""

from typing import Any, Dict, Mapping, Type


class SchemaValidationError(ValueError):
    """Raised when an LLM response does not match the expected contract."""

    def __init__(self, stage: str, message: str, field: str = ""):
        self.stage = stage
        self.field = field
        location = f" field={field}" if field else ""
        super().__init__(f"{stage} 输出校验失败{location}: {message}")


MAP_FIELD_TYPES: Dict[str, Type[Any]] = {
    "summary": str,
    "vibe": str,
    "active_members": list,
    "inactive_members": list,
    "events": list,
    "memes_born": list,
    "memes_died": list,
    "mvp": str,
    "characters": dict,
    "relations": list,
}

REDUCE_FIELD_TYPES: Dict[str, Type[Any]] = {
    "style_config": dict,
    "keywords": list,
    "portrait": str,
    "timeline": str,
    "quarterly_review": str,
    "roasts": str,
    "awards": str,
    "anime_theater": str,
    "moments": str,
    "essay": str,
}


def _validate_object(
    value: Any,
    stage: str,
    field_types: Mapping[str, Type[Any]],
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(stage, "顶层必须是 JSON 对象，而不是列表、字符串或其他类型")

    for field, expected_type in field_types.items():
        if field not in value:
            raise SchemaValidationError(stage, "缺少必需字段", field)
        field_value = value[field]
        if not isinstance(field_value, expected_type):
            raise SchemaValidationError(
                stage,
                f"类型必须是 {expected_type.__name__}，实际是 {type(field_value).__name__}",
                field,
            )

    list_item_types = {
        "active_members": (str,),
        "inactive_members": (str,),
        "events": (str, dict),
        "memes_born": (str, dict),
        "memes_died": (str, dict),
        "relations": (str, dict),
        "keywords": (str,),
    }
    for field, allowed_types in list_item_types.items():
        if field not in value:
            continue
        invalid_items = [
            item for item in value[field]
            if not isinstance(item, allowed_types)
        ]
        if invalid_items:
            expected = "/".join(item_type.__name__ for item_type in allowed_types)
            raise SchemaValidationError(
                stage,
                f"列表元素必须是 {expected}",
                field,
            )

    characters = value.get("characters")
    if stage == "Map" and isinstance(characters, dict):
        for key, character in characters.items():
            if not isinstance(key, str) or not isinstance(character, str):
                raise SchemaValidationError(
                    stage,
                    "characters 的键和值必须是字符串",
                    "characters",
                )

    style_config = value.get("style_config")
    if stage == "Reduce" and isinstance(style_config, dict):
        for key, style_value in style_config.items():
            if not isinstance(key, str) or not isinstance(style_value, str):
                raise SchemaValidationError(
                    stage,
                    "style_config 的键和值必须是字符串",
                    "style_config",
                )

    return dict(value)


def validate_map_result(value: Any) -> Dict[str, Any]:
    """Validate and return one Map result."""
    return _validate_object(value, "Map", MAP_FIELD_TYPES)


def validate_reduce_result(value: Any) -> Dict[str, Any]:
    """Validate and return one Reduce result."""
    return _validate_object(value, "Reduce", REDUCE_FIELD_TYPES)
