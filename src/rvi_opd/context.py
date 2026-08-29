from __future__ import annotations

import hashlib
import json
from typing import Sequence


def _validated_context_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise ValueError("token_ids must be a sequence of non-negative integer token IDs")
    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise ValueError("token_ids must be a sequence of non-negative integer token IDs") from exc
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in values
    ):
        raise ValueError("token_ids must contain only non-negative integers")
    return values


def context_hash(token_ids: Sequence[int], tokenizer_sha256: str) -> str:
    """Hash an exact token context without coercing floats/strings/bools to IDs."""

    values = _validated_context_token_ids(token_ids)
    if not isinstance(tokenizer_sha256, str) or not tokenizer_sha256:
        raise ValueError("tokenizer_sha256 must be a non-empty string")
    payload = {
        "token_ids": list(values),
        "tokenizer_sha256": tokenizer_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_detached_contract(
    original_token_ids: Sequence[int],
    detached_post_leg_token_ids: Sequence[int],
    normal_post_leg_token_ids: Sequence[int],
    tokenizer_sha256: str,
) -> None:
    original = context_hash(original_token_ids, tokenizer_sha256)
    detached = context_hash(detached_post_leg_token_ids, tokenizer_sha256)
    normal = context_hash(normal_post_leg_token_ids, tokenizer_sha256)
    if detached != original:
        raise AssertionError("detached post-leg context does not equal the original context")
    if normal == original:
        raise AssertionError("normal bridge did not change the post-leg context")


def assert_rollback_contract(
    original_token_ids: Sequence[int], rollback_token_ids: Sequence[int], tokenizer_sha256: str
) -> None:
    if context_hash(original_token_ids, tokenizer_sha256) != context_hash(
        rollback_token_ids, tokenizer_sha256
    ):
        raise AssertionError("gate rollback did not restore the original context")
