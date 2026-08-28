from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_EXPERIMENT_FIELDS = {
    "id",
    "question",
    "split",
    "arms",
    "primary_endpoints",
    "audit_invariants",
}

REQUIRED_STRING_FIELDS = ("id", "question", "split")
REQUIRED_STRING_LIST_FIELDS = ("arms", "primary_endpoints", "audit_invariants")

PROBABILITY_FIELD_NAMES = {
    "acceptance_rate",
    "error_rate",
    "p",
    "p_hat",
    "pass_rate",
    "probability",
    "probabilities",
    "quantile",
    "quantiles",
    "clip_ratio",
    "relative_tolerance",
    "repetition_rate",
    "state_bands",
    "success_rate",
    "top_p",
    "trigger_rate",
}
PROBABILITY_FIELD_SUFFIXES = (
    "_fraction",
    "_fractions",
    "_probability",
    "_probabilities",
    "_quantile",
    "_quantiles",
    "_acceptance_rate",
    "_error_rate",
    "_pass_rate",
    "_repetition_rate",
    "_success_rate",
    "_trigger_rate",
    "_top_p",
)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("top-level config must be an object")
    return payload


def _is_probability_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in PROBABILITY_FIELD_NAMES or normalized.endswith(
        PROBABILITY_FIELD_SUFFIXES
    )


def _validate_value(
    value: Any,
    field_path: str,
    errors: List[str],
    probability_context: bool = False,
) -> None:
    if value is None:
        errors.append(f"{field_path} must not be null")
        return
    if isinstance(value, bool):
        if probability_context:
            errors.append(f"{field_path} must be numeric and in [0, 1]")
        return
    if isinstance(value, str):
        if probability_context:
            errors.append(f"{field_path} must be numeric and in [0, 1]")
        elif value == "":
            errors.append(f"{field_path} must be a non-empty string")
        return
    if isinstance(value, int):
        if value < 0:
            errors.append(f"{field_path} must be non-negative")
        elif probability_context and value > 1:
            errors.append(f"{field_path} must be in [0, 1]")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{field_path} must be finite")
        elif value < 0:
            errors.append(f"{field_path} must be non-negative")
        elif probability_context and value > 1:
            errors.append(f"{field_path} must be in [0, 1]")
        return
    if isinstance(value, list):
        if not value:
            errors.append(f"{field_path} must be a non-empty list")
            return
        for index, item in enumerate(value):
            _validate_value(item, f"{field_path}[{index}]", errors, probability_context)
        return
    if isinstance(value, dict):
        if not value:
            errors.append(f"{field_path} must be a non-empty object")
            return
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{field_path} contains an empty or non-string field name")
                continue
            child_path = f"{field_path}.{key}"
            _validate_value(
                item,
                child_path,
                errors,
                probability_context or _is_probability_field(key),
            )
        return
    errors.append(f"{field_path} has unsupported type {type(value).__name__}")


def _validate_required_string_lists(
    payload: Dict[str, Any], source: str, errors: List[str]
) -> None:
    for field in REQUIRED_STRING_LIST_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, list):
            errors.append(f"{source}: {field} must be a non-empty list")
            continue
        seen = set()
        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(f"{source}: {field}[{index}] must be a non-empty string")
                continue
            if item and not item.strip():
                errors.append(f"{source}: {field}[{index}] must be a non-empty string")
            if item in seen:
                errors.append(f"{source}: {field} contains duplicate value {item!r}")
            seen.add(item)


def validate_experiment_config(payload: Any, source: str = "<memory>") -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"{source}: top-level config must be an object"]

    missing = REQUIRED_EXPERIMENT_FIELDS - set(payload)
    if missing:
        errors.append(f"{source}: missing fields: {', '.join(sorted(missing))}")

    for field in REQUIRED_STRING_FIELDS:
        if field in payload:
            value = payload[field]
            if not isinstance(value, str) or (value and not value.strip()):
                errors.append(f"{source}: {field} must be a non-empty string")
    _validate_required_string_lists(payload, source, errors)

    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            errors.append(f"{source}: top-level config contains an empty or non-string field name")
            continue
        _validate_value(value, f"{source}: {key}", errors, _is_probability_field(key))
    return errors


def validate_config_paths(paths: Iterable[Path]) -> List[str]:
    errors: List[str] = []
    seen_ids: Dict[str, Path] = {}
    for path in paths:
        try:
            payload = load_config(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        errors.extend(validate_experiment_config(payload, str(path)))
        experiment_id = payload.get("id")
        if isinstance(experiment_id, str) and experiment_id.strip():
            if experiment_id in seen_ids:
                errors.append(
                    f"{path}: duplicate experiment id {experiment_id!r}; "
                    f"first declared in {seen_ids[experiment_id]}"
                )
            else:
                seen_ids[experiment_id] = path
    return errors
