"""Dependency-free prompt and context-budget contracts for the Qwen3 tracks.

The Qwen3 teacher and base student expose different chat-template behavior
when ``enable_thinking=False`` is passed to ``apply_chat_template``.  The
experiment therefore uses a project-owned, deterministic serializer and
checks its tokenized rendering in the C0 compatibility fixture before any
training or benchmark generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


CANONICAL_SERIALIZER_VERSION = "rvi_opd_non_thinking_v1"
C0_ARTIFACT_VERSION = "rvi-c0-v1"
DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
MEDICAL_SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Provide a clear, complete, and safe answer "
    "while stating uncertainty and escalation conditions when relevant."
)

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
_QWEN_CONTROL_MARKER = re.compile(r"<\|[^<>\r\n]{1,128}\|>")
_RESERVED_XML_MARKER = re.compile(
    r"</?(?:think|tool_call|tool_response|function_call|assistant|system|user)"
    r"(?:\s[^<>]*)?>",
    flags=re.IGNORECASE,
)

# Keep an explicit inventory for audit artifacts, while rejecting every
# ``<|...|>`` control marker in untrusted prompt fields so a newly introduced
# Qwen marker cannot silently bypass this list.
QWEN_RESERVED_SPECIAL_MARKERS = (
    "<|endoftext|>",
    _IM_START,
    _IM_END,
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

QWEN3_VOCAB_SHA256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
QWEN3_VOCAB_SIZE = 151936
QWEN3_TOKENIZER_LENGTH = 151669
QWEN3_SPECIAL_TOKEN_MAP = {
    "im_start": 151644,
    "im_end": 151645,
    "think_start": 151667,
    "think_end": 151668,
    "eos": (151645, 151643),
    "pad": 151643,
}
# The pinned tokenizer reserves the contiguous tail from ``<|endoftext|>``
# through ``</think>``.  Some IDs in this interval are multimodal or future
# control markers and are intentionally absent from the minimal semantic map
# above, but none may constitute an entire supervised target.
QWEN3_RESERVED_SPECIAL_TOKEN_IDS = tuple(range(151643, 151669))

E1_TRAINING_PROFILE = "e1_training"
E1_EVALUATION_PROFILE = "e1_evaluation"
E2_TRAINING_PROFILE = "e2_training"
E2_EVALUATION_PROFILE = "e2_evaluation"
DEFAULT_GENERATION_PROFILE = E1_EVALUATION_PROFILE


def _profile(
    max_input_tokens: int, max_new_tokens: int, runtime_context_limit: int
) -> Dict[str, object]:
    return {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "stop_token_ids": (151645, 151643),
        "pad_token_id": 151643,
        "num_return_sequences": 1,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "max_response_tokens": max_new_tokens,
        "runtime_context_limit": runtime_context_limit,
        "max_model_length": runtime_context_limit,
        "eos_reserve": 0,
    }


_STRICT_GENERATION_PROFILES = {
    E1_TRAINING_PROFILE: _profile(2048, 16384, 34817),
    E1_EVALUATION_PROFILE: _profile(2048, 32768, 34817),
    E2_TRAINING_PROFILE: _profile(2048, 8192, 40960),
    E2_EVALUATION_PROFILE: _profile(36864, 4096, 40960),
}


def _validated_token_ids(
    token_ids: Sequence[int], name: str, *, allow_empty: bool
) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer token IDs")
    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integer token IDs") from exc
    if (not allow_empty and not values) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in values
    ):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise ValueError(f"{name} must be a {qualifier}sequence of non-negative integers")
    return values


def _validated_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA256")
    return value


def _validated_revision(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be an immutable lowercase 40-hex revision")
    return value


def _validated_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_special_token_map(
    special_token_map: object,
) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    if isinstance(special_token_map, Mapping):
        raw_items = tuple(special_token_map.items())
    elif isinstance(special_token_map, (str, bytes)):
        raise ValueError("special_token_map must be a non-empty mapping or normalized pairs")
    else:
        try:
            raw_items = tuple(special_token_map)
        except TypeError as exc:
            raise ValueError(
                "special_token_map must be a non-empty mapping or normalized pairs"
            ) from exc
    if not raw_items:
        raise ValueError("special_token_map must be non-empty")
    normalized = []
    for raw_item in raw_items:
        if not isinstance(raw_item, (tuple, list)) or len(raw_item) != 2:
            raise ValueError("special_token_map normalized entries must be name/value pairs")
        name, raw_value = raw_item
        if not isinstance(name, str) or not name.strip():
            raise ValueError("special-token names must be non-empty strings")
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            token_ids = _validated_token_ids(
                (raw_value,), f"special_token_map[{name!r}]", allow_empty=False
            )
        else:
            token_ids = _validated_token_ids(
                raw_value, f"special_token_map[{name!r}]", allow_empty=False
            )
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(f"special_token_map[{name!r}] contains duplicate IDs")
        normalized.append((name, token_ids))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("special_token_map contains duplicate names")
    return tuple(sorted(normalized))


def _special_token_map_as_dict(
    normalized: Sequence[Tuple[str, Sequence[int]]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name, raw_ids in normalized:
        token_ids = tuple(raw_ids)
        result[name] = token_ids[0] if len(token_ids) == 1 else list(token_ids)
    return result


def _assert_safe_prompt_field(value: str, name: str) -> None:
    if _QWEN_CONTROL_MARKER.search(value) or _RESERVED_XML_MARKER.search(value):
        raise ValueError(f"{name} must not contain Qwen control, role, thinking, or tool markers")


@dataclass(frozen=True)
class RenderedPrompt:
    """A serialized prompt plus its content hash.

    ``token_ids`` is intentionally optional: the dependency-free core can
    freeze and compare the text hash, while a model-backed C0 fixture can add
    the exact tokenizer output and compare teacher/student IDs.
    """

    text: str
    serializer_version: str = CANONICAL_SERIALIZER_VERSION
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("rendered prompt text must be a non-empty string")
        if not isinstance(self.serializer_version, str) or not self.serializer_version.strip():
            raise ValueError("serializer_version must be a non-empty string")
        object.__setattr__(
            self,
            "token_ids",
            _validated_token_ids(self.token_ids, "token_ids", allow_empty=True),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class C0Artifact:
    """Immutable compatibility evidence required before a production run.

    The artifact binds the effective vocabulary, tokenizer cardinality,
    special-token semantics, canonical serializer, exact rendered prompt IDs,
    model revisions and strict generation profile.  Construction validates
    types; :meth:`assert_production_ready` additionally enforces the locked
    Qwen3 experiment values and teacher/student token equality.
    """

    teacher_revision: str
    student_revision: str
    vocab_sha256: str
    vocab_size: int
    tokenizer_length: int
    special_token_map: Mapping[str, object]
    serializer_version: str
    rendered_prompt_sha256: str
    teacher_rendered_token_ids: tuple[int, ...]
    student_rendered_token_ids: tuple[int, ...]
    generation_profile: str
    schema_version: str = C0_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "serializer_version",
            "generation_profile",
            "schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "teacher_revision",
            _validated_revision(self.teacher_revision, "teacher_revision"),
        )
        object.__setattr__(
            self,
            "student_revision",
            _validated_revision(self.student_revision, "student_revision"),
        )
        object.__setattr__(
            self, "vocab_sha256", _validated_sha256(self.vocab_sha256, "vocab_sha256")
        )
        object.__setattr__(
            self,
            "rendered_prompt_sha256",
            _validated_sha256(self.rendered_prompt_sha256, "rendered_prompt_sha256"),
        )
        object.__setattr__(
            self, "vocab_size", _validated_positive_integer(self.vocab_size, "vocab_size")
        )
        object.__setattr__(
            self,
            "tokenizer_length",
            _validated_positive_integer(self.tokenizer_length, "tokenizer_length"),
        )
        if self.tokenizer_length > self.vocab_size:
            raise ValueError("tokenizer_length cannot exceed vocab_size")
        object.__setattr__(
            self,
            "special_token_map",
            _normalize_special_token_map(self.special_token_map),
        )
        object.__setattr__(
            self,
            "teacher_rendered_token_ids",
            _validated_token_ids(
                self.teacher_rendered_token_ids,
                "teacher_rendered_token_ids",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "student_rendered_token_ids",
            _validated_token_ids(
                self.student_rendered_token_ids,
                "student_rendered_token_ids",
                allow_empty=False,
            ),
        )

    @property
    def artifact_sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_artifact_sha256: bool = False) -> Dict[str, object]:
        if not isinstance(include_artifact_sha256, bool):
            raise ValueError("include_artifact_sha256 must be boolean")
        payload = {
            "schema_version": self.schema_version,
            "teacher_revision": self.teacher_revision,
            "student_revision": self.student_revision,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": self.vocab_size,
            "tokenizer_length": self.tokenizer_length,
            "special_token_map": _special_token_map_as_dict(self.special_token_map),
            "serializer_version": self.serializer_version,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "teacher_rendered_token_ids": list(self.teacher_rendered_token_ids),
            "student_rendered_token_ids": list(self.student_rendered_token_ids),
            "generation_profile": self.generation_profile,
        }
        if include_artifact_sha256:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "C0Artifact":
        if not isinstance(payload, Mapping):
            raise ValueError("C0 artifact payload must be a mapping")
        required = {
            "teacher_revision",
            "student_revision",
            "vocab_sha256",
            "vocab_size",
            "tokenizer_length",
            "special_token_map",
            "serializer_version",
            "rendered_prompt_sha256",
            "teacher_rendered_token_ids",
            "student_rendered_token_ids",
            "generation_profile",
        }
        allowed = required | {"schema_version", "artifact_sha256"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("C0 artifact is missing fields: " + ", ".join(missing))
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise ValueError("C0 artifact has unknown fields: " + ", ".join(unknown))
        claimed_sha256 = payload.get("artifact_sha256")
        if claimed_sha256 is not None:
            _validated_sha256(claimed_sha256, "artifact_sha256")
        artifact = cls(
            teacher_revision=payload["teacher_revision"],
            student_revision=payload["student_revision"],
            vocab_sha256=payload["vocab_sha256"],
            vocab_size=payload["vocab_size"],
            tokenizer_length=payload["tokenizer_length"],
            special_token_map=payload["special_token_map"],
            serializer_version=payload["serializer_version"],
            rendered_prompt_sha256=payload["rendered_prompt_sha256"],
            teacher_rendered_token_ids=payload["teacher_rendered_token_ids"],
            student_rendered_token_ids=payload["student_rendered_token_ids"],
            generation_profile=payload["generation_profile"],
            schema_version=payload.get("schema_version", C0_ARTIFACT_VERSION),
        )
        if claimed_sha256 is not None and claimed_sha256 != artifact.artifact_sha256:
            raise ValueError("C0 artifact SHA256 does not match its unsigned payload")
        return artifact

    def assert_production_ready(self) -> None:
        if self.schema_version != C0_ARTIFACT_VERSION:
            raise ValueError(f"C0 artifact schema must be {C0_ARTIFACT_VERSION!r} for production")
        if self.vocab_sha256 != QWEN3_VOCAB_SHA256:
            raise ValueError(
                "C0 artifact vocabulary hash does not match the locked Qwen3 vocabulary"
            )
        if self.vocab_size != QWEN3_VOCAB_SIZE:
            raise ValueError("C0 artifact vocab_size does not match the locked Qwen3 value")
        if self.tokenizer_length != QWEN3_TOKENIZER_LENGTH:
            raise ValueError("C0 artifact tokenizer_length does not match the locked Qwen3 value")
        expected_map = _normalize_special_token_map(QWEN3_SPECIAL_TOKEN_MAP)
        if self.special_token_map != expected_map:
            raise ValueError("C0 artifact special-token map does not match the locked Qwen3 map")
        for _, token_ids in self.special_token_map:
            if any(token_id >= self.tokenizer_length for token_id in token_ids):
                raise ValueError("C0 artifact special-token ID is outside tokenizer_length")
        if self.serializer_version != CANONICAL_SERIALIZER_VERSION:
            raise ValueError("C0 artifact does not use the canonical serializer version")
        if self.generation_profile not in _STRICT_GENERATION_PROFILES:
            raise ValueError("C0 artifact generation_profile is not a locked profile")
        assert_tokenizer_alignment(self.teacher_rendered_token_ids, self.student_rendered_token_ids)
        if any(token_id >= self.tokenizer_length for token_id in self.teacher_rendered_token_ids):
            raise ValueError("rendered prompt token ID is outside tokenizer_length")


def build_c0_artifact(
    *,
    rendered_prompt: RenderedPrompt,
    teacher_rendered_token_ids: Sequence[int],
    student_rendered_token_ids: Sequence[int],
    teacher_revision: str,
    student_revision: str,
    vocab_sha256: str,
    vocab_size: int,
    tokenizer_length: int,
    special_token_map: Mapping[str, object],
    generation_profile: str,
) -> C0Artifact:
    if not isinstance(rendered_prompt, RenderedPrompt):
        raise ValueError("rendered_prompt must be a RenderedPrompt")
    assert_no_unintended_thinking(rendered_prompt.text)
    if rendered_prompt.token_ids:
        assert_tokenizer_alignment(rendered_prompt.token_ids, teacher_rendered_token_ids)
        assert_tokenizer_alignment(rendered_prompt.token_ids, student_rendered_token_ids)
    artifact = C0Artifact(
        teacher_revision=teacher_revision,
        student_revision=student_revision,
        vocab_sha256=vocab_sha256,
        vocab_size=vocab_size,
        tokenizer_length=tokenizer_length,
        special_token_map=special_token_map,
        serializer_version=rendered_prompt.serializer_version,
        rendered_prompt_sha256=rendered_prompt.sha256,
        teacher_rendered_token_ids=tuple(teacher_rendered_token_ids),
        student_rendered_token_ids=tuple(student_rendered_token_ids),
        generation_profile=generation_profile,
    )
    artifact.assert_production_ready()
    return artifact


def assert_c0_production_ready(artifact: C0Artifact) -> None:
    if not isinstance(artifact, C0Artifact):
        raise ValueError("artifact must be a C0Artifact")
    artifact.assert_production_ready()


def render_non_thinking_prompt(
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> RenderedPrompt:
    """Render one canonical Qwen-style prompt without an implicit think block.

    The serializer deliberately does not call a model-specific chat-template
    helper or pass ``enable_thinking``.  This avoids the Qwen3 base student's
    empty ``<think>`` insertion and makes teacher/student rendering a tested
    project contract.  A downstream adapter must still verify that the
    tokenizer maps both model revisions to the same IDs.
    """

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    _assert_safe_prompt_field(user_prompt, "user_prompt")
    _assert_safe_prompt_field(system_prompt, "system_prompt")
    text = (
        f"{_IM_START}system\n{system_prompt}{_IM_END}\n"
        f"{_IM_START}user\n{user_prompt}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )
    assert_no_unintended_thinking(text)
    return RenderedPrompt(text=text)


def assert_no_unintended_thinking(text: str) -> None:
    """Reject hidden/non-empty reasoning blocks in a rendered prompt/output."""

    if not isinstance(text, str):
        raise ValueError("rendered text must be a string")
    blocks = _THINK_BLOCK.findall(text)
    if blocks:
        raise ValueError("rendered prompt contains an unintended <think> block")
    lowered = text.lower()
    if "<think>" in lowered or "</think>" in lowered:
        raise ValueError("rendered prompt contains an unmatched think marker")


def assert_tokenizer_alignment(
    teacher_token_ids: Sequence[int], student_token_ids: Sequence[int]
) -> None:
    """Require exact canonical rendered IDs, not merely equal vocab sizes."""

    teacher = _validated_token_ids(teacher_token_ids, "teacher_token_ids", allow_empty=False)
    student = _validated_token_ids(student_token_ids, "student_token_ids", allow_empty=False)
    if teacher != student:
        raise ValueError(
            "teacher and student canonical prompt token IDs differ; "
            "freeze a compatible serializer before running the experiment"
        )


def assert_context_budget(
    rendered_input_tokens: int,
    max_new_tokens: int,
    runtime_context_limit: int,
    eos_reserve: int = 0,
    *,
    max_input_tokens: Optional[int] = None,
) -> None:
    """Fail before generation when either the input cap or total context is exceeded.

    ``max_input_tokens`` is checked independently of the summed context
    budget.  This matters for E2, whose locked request cap is 36,864 input
    tokens even if a caller proposes fewer than the locked 4,096 new tokens.
    """

    values = (rendered_input_tokens, max_new_tokens, runtime_context_limit, eos_reserve)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("context-budget values must be integers")
    if max_input_tokens is not None and (
        isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int)
    ):
        raise ValueError("max_input_tokens must be an integer when provided")
    if (
        rendered_input_tokens < 0
        or max_new_tokens < 0
        or runtime_context_limit <= 0
        or eos_reserve < 0
        or (max_input_tokens is not None and max_input_tokens < 0)
    ):
        raise ValueError("context-budget values are out of range")
    if max_input_tokens is not None and rendered_input_tokens > max_input_tokens:
        raise ValueError("rendered input exceeds the independently locked max_input_tokens cap")
    if rendered_input_tokens + max_new_tokens + eos_reserve > runtime_context_limit:
        raise ValueError(
            "rendered input plus max_new_tokens plus eos_reserve exceeds the runtime context limit; "
            "benchmark truncation is not permitted"
        )


def build_generation_contract(
    *,
    profile: str = DEFAULT_GENERATION_PROFILE,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    stop_token_ids: Optional[Sequence[int]] = None,
    pad_token_id: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    max_response_tokens: Optional[int] = None,
    runtime_context_limit: Optional[int] = None,
    max_model_length: Optional[int] = None,
    eos_reserve: Optional[int] = None,
) -> Dict[str, object]:
    """Return one immutable E1/E2 generation profile and reject drift.

    Model-card defaults are deliberately not inherited. ``top_k=0`` is the
    project-level representation of a disabled top-k filter; adapters must map
    it to the backend's documented disabled value (Relay/verl uses ``-1``) and
    record that resolved payload. ``max_new_tokens`` includes the token that
    causes one of the stop IDs; no extra hidden EOS reserve is added here.  An
    explicit value is accepted only when it exactly equals the selected locked
    profile, so this helper cannot be used to smuggle model-card defaults or a
    shorter/longer response cap into a preregistered run.
    """

    if not isinstance(profile, str) or profile not in _STRICT_GENERATION_PROFILES:
        allowed = ", ".join(sorted(_STRICT_GENERATION_PROFILES))
        raise ValueError(f"unknown strict generation profile; expected one of: {allowed}")
    expected = {
        key: list(value) if key == "stop_token_ids" else value
        for key, value in _STRICT_GENERATION_PROFILES[profile].items()
    }

    if max_new_tokens is not None and max_response_tokens is not None:
        if max_new_tokens != max_response_tokens:
            raise ValueError("max_new_tokens and max_response_tokens aliases disagree")
    requested_new_tokens = max_new_tokens if max_new_tokens is not None else max_response_tokens
    if runtime_context_limit is not None and max_model_length is not None:
        if runtime_context_limit != max_model_length:
            raise ValueError("runtime_context_limit and max_model_length aliases disagree")
    requested_context_limit = (
        runtime_context_limit if runtime_context_limit is not None else max_model_length
    )

    explicit: Dict[str, object] = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": pad_token_id,
        "num_return_sequences": num_return_sequences,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": requested_new_tokens,
        "runtime_context_limit": requested_context_limit,
        "eos_reserve": eos_reserve,
    }
    if stop_token_ids is not None:
        explicit["stop_token_ids"] = list(
            _validated_token_ids(stop_token_ids, "stop_token_ids", allow_empty=False)
        )
    else:
        explicit["stop_token_ids"] = None

    for key, value in explicit.items():
        if value is None:
            continue
        if key == "do_sample":
            if not isinstance(value, bool):
                raise ValueError("do_sample must be boolean")
        elif key in {"temperature", "top_p"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{key} must be a finite number")
            value = float(value)
        elif key == "stop_token_ids":
            pass
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if value != expected[key]:
            raise ValueError(
                f"{key}={value!r} drifts from locked {profile} value {expected[key]!r}"
            )

    contract = {"profile": profile, **expected}
    assert_strict_generation_contract(contract, profile)
    return contract


def assert_strict_generation_contract(
    contract: Mapping[str, object], profile: Optional[str] = None
) -> None:
    """Validate a serialized generation request against a named frozen profile."""

    if not isinstance(contract, Mapping):
        raise ValueError("generation contract must be a mapping")
    resolved_profile = contract.get("profile") if profile is None else profile
    if not isinstance(resolved_profile, str) or resolved_profile not in _STRICT_GENERATION_PROFILES:
        raise ValueError("generation contract has no recognized strict profile")
    if "profile" not in contract or contract["profile"] != resolved_profile:
        raise ValueError("generation contract profile metadata is missing or inconsistent")
    expected = _STRICT_GENERATION_PROFILES[resolved_profile]
    unknown = sorted(set(contract).difference({"profile", *expected}))
    if unknown:
        raise ValueError("generation contract has unregistered fields: " + ", ".join(unknown))
    for key, expected_value in expected.items():
        if key not in contract:
            raise ValueError(f"generation contract is missing locked field {key!r}")
        value = contract[key]
        if key == "do_sample":
            if not isinstance(value, bool):
                raise ValueError("generation contract field 'do_sample' must be boolean")
        elif key == "stop_token_ids":
            value = _validated_token_ids(value, "stop_token_ids", allow_empty=False)
        elif key in {"temperature", "top_p"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"generation contract field {key!r} must be finite numeric")
            value = float(value)
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"generation contract field {key!r} must be an integer")
        if value != expected_value:
            raise ValueError(f"generation contract field {key!r} drifts from {resolved_profile}")


def assert_generation_request(contract: Mapping[str, object], rendered_input_tokens: int) -> None:
    """Validate a strict contract and the actual rendered input before dispatch."""

    assert_strict_generation_contract(contract)
    assert_context_budget(
        rendered_input_tokens,
        contract["max_new_tokens"],
        contract["runtime_context_limit"],
        contract["eos_reserve"],
        max_input_tokens=contract["max_input_tokens"],
    )


def build_target_text(complex_cot: str, response: str) -> str:
    """Map medical-o1 fields to the canonical supervised assistant target."""

    if not isinstance(complex_cot, str) or not complex_cot.strip():
        raise ValueError("complex_cot must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")
    return complex_cot.rstrip() + "\n\n" + response.lstrip()


def supervision_mask(
    sequence_length: int,
    target_start: int,
    target_end: int,
    *,
    token_ids: Optional[Sequence[int]] = None,
    special_token_ids: Sequence[int] = QWEN3_RESERVED_SPECIAL_TOKEN_IDS,
    allow_unverified_target_tokens: bool = False,
) -> Tuple[int, ...]:
    """Create a deterministic 0/1 mask for assistant target positions.

    ``target_end`` is exclusive.  System/user messages and chat markers are
    represented by zeros.  Production callers must provide ``token_ids`` so
    the target can be rejected when it is entirely made of Qwen special
    markers.  The explicit ``allow_unverified_target_tokens`` escape hatch is
    retained only for generic, non-production mask construction.
    """

    values = (sequence_length, target_start, target_end)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("supervision-mask bounds must be integers")
    if sequence_length < 0 or not 0 <= target_start < target_end <= sequence_length:
        raise ValueError("supervision-mask bounds are out of range")
    if not isinstance(allow_unverified_target_tokens, bool):
        raise ValueError("allow_unverified_target_tokens must be boolean")
    if token_ids is None:
        if not allow_unverified_target_tokens:
            raise ValueError(
                "token_ids are required to prove that the supervised target is not all special"
            )
    else:
        sequence_ids = _validated_token_ids(token_ids, "token_ids", allow_empty=False)
        if len(sequence_ids) != sequence_length:
            raise ValueError("token_ids length must equal sequence_length")
        specials = set(
            _validated_token_ids(special_token_ids, "special_token_ids", allow_empty=False)
        )
        target_ids = sequence_ids[target_start:target_end]
        if all(token_id in specials for token_id in target_ids):
            raise ValueError("supervised target must contain at least one non-special token")
    return tuple(1 if target_start <= index < target_end else 0 for index in range(sequence_length))
