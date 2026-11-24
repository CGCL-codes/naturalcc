from __future__ import annotations

import re
from typing import Any


USER_INPUT_TOOL_NAME = "user_ask"
_ANSWER_TYPES = ("text", "choice", "confirm")
_MAX_QUESTION_CHARS = 1_000
_MAX_ANSWER_CHARS = 4_000
_MIN_OPTIONS = 2
_MAX_OPTIONS = 8
_MAX_LABEL_CHARS = 200
_OPTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$"
_UNSAFE_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")


class HumanInputValidationError(ValueError):
    """The question or answer does not conform to the user-input contract."""


def user_input_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": USER_INPUT_TOOL_NAME,
            "description": (
                "Ask the user one necessary clarification or decision, then wait for their answer. "
                "Use text for free input, choice for two to eight options, or confirm for a yes/no decision. "
                "Never ask for credentials. An answer does not grant tool permissions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 1, "maxLength": _MAX_QUESTION_CHARS},
                    "answer_type": {"type": "string", "enum": list(_ANSWER_TYPES)},
                    "options": {
                        "type": "array",
                        "description": "Required for choice. Omit or leave empty for text and confirm.",
                        "maxItems": _MAX_OPTIONS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "pattern": _OPTION_ID_PATTERN},
                                "label": {"type": "string", "minLength": 1, "maxLength": _MAX_LABEL_CHARS},
                            },
                            "required": ["id", "label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["question", "answer_type"],
                "additionalProperties": False,
            },
        },
    }


def _bounded_text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit or not value.strip() or _UNSAFE_TEXT.search(value):
        raise HumanInputValidationError(f"{field} must be non-empty text of at most {limit} characters.")
    return value.strip()


def validate_question(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict) or set(args) - {"question", "answer_type", "options"}:
        raise HumanInputValidationError("The question contains invalid fields.")
    question = _bounded_text(args.get("question"), "Question", _MAX_QUESTION_CHARS)
    answer_type = args.get("answer_type")
    if answer_type not in _ANSWER_TYPES:
        raise HumanInputValidationError("Answer type must be text, choice, or confirm.")
    raw_options = args.get("options", [])
    if not isinstance(raw_options, list):
        raise HumanInputValidationError("Question options must be an array.")
    if answer_type != "choice":
        if raw_options:
            raise HumanInputValidationError("Only choice questions can include options.")
        return {"question": question, "answer_type": answer_type, "options": []}
    if not _MIN_OPTIONS <= len(raw_options) <= _MAX_OPTIONS:
        raise HumanInputValidationError("A choice question requires two to eight options.")

    options = []
    ids = set()
    for option in raw_options:
        if not isinstance(option, dict) or set(option) != {"id", "label"}:
            raise HumanInputValidationError("Each option must contain an id and a label.")
        option_id = option["id"]
        if not isinstance(option_id, str) or not re.fullmatch(_OPTION_ID_PATTERN, option_id) or option_id in ids:
            raise HumanInputValidationError("Each option must have a unique short identifier.")
        ids.add(option_id)
        options.append({"id": option_id, "label": _bounded_text(option["label"], "Option label", _MAX_LABEL_CHARS)})
    return {"question": question, "answer_type": answer_type, "options": options}


def validate_answer(pending: dict[str, Any], value: Any) -> str | bool:
    answer_type = pending.get("answer_type")
    if answer_type == "text":
        return _bounded_text(value, "Answer", _MAX_ANSWER_CHARS)
    if answer_type == "confirm":
        if type(value) is not bool:
            raise HumanInputValidationError("Confirmation requires a yes or no answer.")
        return value
    if answer_type == "choice":
        options = pending.get("options")
        if not isinstance(value, str) or not isinstance(options, list) or not any(
            isinstance(option, dict) and option.get("id") == value for option in options
        ):
            raise HumanInputValidationError("Answer must select an available option.")
        return value
    raise HumanInputValidationError("The requested answer type is invalid.")
