from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .models import RawStateSignal, StateSignal

Token = Hashable
Distribution = Mapping[Token, float]

_FROZEN_SCALE_EPSILON = 1e-12


@dataclass(frozen=True)
class FrozenScaleArtifact:
    """D1-fitted raw-signal anchors used by every production routing request."""

    q_low: float
    q_high: float
    divergence_q_low: float
    divergence_q_high: float
    compatibility_q_low: float
    compatibility_q_high: float
    vocabulary_sha256: str = ""
    code_revision: str = ""
    scale_schema_version: str = "rvi-frozen-scale-v1"

    def __post_init__(self) -> None:
        numeric_names = (
            "q_low",
            "q_high",
            "divergence_q_low",
            "divergence_q_high",
            "compatibility_q_low",
            "compatibility_q_high",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not 0 <= self.q_low < self.q_high <= 1:
            raise ValueError("scale quantiles must satisfy 0 <= q_low < q_high <= 1")
        if not 0 <= self.divergence_q_low <= self.divergence_q_high:
            raise ValueError("divergence anchors must be ordered and non-negative")
        if not 0 <= self.compatibility_q_low <= self.compatibility_q_high <= 1:
            raise ValueError("compatibility anchors must be ordered probabilities")
        if not isinstance(self.vocabulary_sha256, str):
            raise ValueError("vocabulary_sha256 must be a string")
        if not isinstance(self.code_revision, str):
            raise ValueError("code_revision must be a string")
        if self.scale_schema_version != "rvi-frozen-scale-v1":
            raise ValueError("unsupported frozen-scale schema version")

    def _unsigned_payload(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def artifact_sha256(self) -> str:
        encoded = json.dumps(
            self._unsigned_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        return {**self._unsigned_payload(), "artifact_sha256": self.artifact_sha256}

    @property
    def production_ready(self) -> bool:
        return bool(
            self.divergence_q_low < self.divergence_q_high
            and self.compatibility_q_low < self.compatibility_q_high
            and re.fullmatch(r"[0-9a-f]{64}", self.vocabulary_sha256)
            and re.fullmatch(r"[0-9a-f]{40}", self.code_revision)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenScaleArtifact":
        expected_hash = payload.get("artifact_sha256")
        field_names = {field.name for field in fields(cls)}
        unknown = set(payload) - field_names - {"artifact_sha256"}
        if unknown:
            raise ValueError("unknown frozen-scale fields: " + ", ".join(sorted(unknown)))
        try:
            artifact = cls(**{key: payload[key] for key in field_names if key in payload})
        except TypeError as exc:
            raise ValueError(f"invalid frozen-scale artifact: {exc}") from exc
        if expected_hash is not None and expected_hash != artifact.artifact_sha256:
            raise ValueError("frozen-scale artifact SHA256 does not match its contents")
        return artifact


EPISTEMIC_ONSET_PHRASES: Tuple[str, ...] = (
    "Wait",
    "Actually",
    "However",
    "Alternatively",
    "Oops",
    "Wrong",
    "Error",
    "Incorrect",
    "Correction",
    "Sorry",
    "Hmm",
    "Oh",
    "Hold",
    "Pause",
    "Uh",
    "Um",
)

# Relay-OPD's handoff lexicon is intentionally separate from TRD's 16-phrase
# epistemic-mass lexicon. Tokenizer adapters must expand case and leading-space
# variants and persist the resulting IDs.
RELAY_REFLECTION_PHRASES: Tuple[str, ...] = (
    "Wait",
    "But",
    "Hmm",
    "Actually",
    "Hold",
    "However",
    "Yet",
    "Oh",
    "Alternatively",
    "No",
    "Ah",
    "Oops",
    "Well",
)


def _validate_distribution(distribution: Distribution, name: str) -> None:
    if not distribution:
        raise ValueError(f"{name} distribution is empty")
    if any(
        (not math.isfinite(value)) or value < 0 or value > 1.0
        for value in distribution.values()
    ):
        raise ValueError(f"{name} distribution contains invalid probability")
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError(f"{name} distribution has zero mass")
    if total > 1.0 + 1e-4:
        raise ValueError(
            f"{name} values must be absolute full-softmax probabilities, not unnormalized scores"
        )


def _top_k(distribution: Distribution, k: int) -> List[Token]:
    if k <= 0:
        raise ValueError("k must be positive")
    return [
        token
        for token, _ in sorted(
            distribution.items(), key=lambda item: (-item[1], str(item[0]))
        )[:k]
    ]


def _renormalize(distribution: Distribution, support: Iterable[Token]) -> Dict[Token, float]:
    support_list = list(support)
    total = sum(distribution.get(token, 0.0) for token in support_list)
    if total <= 0:
        raise ValueError("distribution has zero mass on requested support")
    return {token: distribution.get(token, 0.0) / total for token in support_list}


def local_support_signal(
    teacher: Distribution,
    student: Distribution,
    k: int = 16,
    epsilon: float = 1e-12,
) -> Tuple[float, float]:
    """Return raw forward-KL disagreement D and teacher coverage C.

    D is KL(T || S) after independently renormalizing teacher and student on
    the union of their top-k supports. C is absolute full-softmax teacher mass
    on the student's top-k support and is not renormalized on the gathered
    subset. Inputs must describe the same vocabulary and context; their values
    are absolute softmax probabilities, even when only a gathered subset is
    materialized.
    """

    _validate_distribution(teacher, "teacher")
    _validate_distribution(student, "student")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if set(teacher) != set(student):
        raise ValueError(
            "teacher and student distributions must contain the same vocabulary keys; "
            "materialize both models' probabilities on the complete top-k union"
        )

    student_support = _top_k(student, k)
    teacher_support = _top_k(teacher, k)
    union: Set[Token] = set(student_support) | set(teacher_support)
    teacher_u = _renormalize(teacher, union)
    student_u = _renormalize(student, union)
    divergence = 0.0
    for token, teacher_prob in teacher_u.items():
        if teacher_prob == 0:
            continue
        divergence += teacher_prob * math.log(
            teacher_prob / max(student_u.get(token, 0.0), epsilon)
        )
    compatibility = sum(teacher.get(token, 0.0) for token in student_support)
    return divergence, compatibility


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def robust_normalize(
    values: Sequence[float], q_low: float = 0.05, q_high: float = 0.95, epsilon: float = 1e-12
) -> List[float]:
    if q_low >= q_high:
        raise ValueError("q_low must be smaller than q_high")
    low = quantile(values, q_low)
    high = quantile(values, q_high)
    denominator = high - low + epsilon
    return [min(1.0, max(0.0, (float(value) - low) / denominator)) for value in values]


def fit_frozen_scale(
    records: Sequence[RawStateSignal],
    q_low: float = 0.05,
    q_high: float = 0.95,
    vocabulary_sha256: str = "",
    code_revision: str = "",
) -> FrozenScaleArtifact:
    """Fit one global raw-D/raw-C transform on the frozen D1 calibration split."""

    if not records:
        raise ValueError("frozen-scale calibration records must not be empty")
    if (
        isinstance(q_low, bool)
        or isinstance(q_high, bool)
        or not isinstance(q_low, (int, float))
        or not isinstance(q_high, (int, float))
        or not math.isfinite(float(q_low))
        or not math.isfinite(float(q_high))
        or not 0 <= q_low < q_high <= 1
    ):
        raise ValueError("scale quantiles must satisfy 0 <= q_low < q_high <= 1")
    return FrozenScaleArtifact(
        q_low=float(q_low),
        q_high=float(q_high),
        divergence_q_low=quantile([record.divergence for record in records], q_low),
        divergence_q_high=quantile([record.divergence for record in records], q_high),
        compatibility_q_low=quantile(
            [record.compatibility for record in records], q_low
        ),
        compatibility_q_high=quantile(
            [record.compatibility for record in records], q_high
        ),
        vocabulary_sha256=vocabulary_sha256,
        code_revision=code_revision,
    )


def _fixed_normalize(value: float, low: float, high: float) -> float:
    normalized = (float(value) - low) / (high - low + _FROZEN_SCALE_EPSILON)
    return min(1.0, max(0.0, normalized))


def _decompose_record(
    record: RawStateSignal, d_norm: float, c_norm: float
) -> StateSignal:
    return StateSignal(
        state_id=record.state_id,
        problem_id=record.problem_id,
        trajectory_id=record.trajectory_id,
        token_index=record.token_index,
        divergence=record.divergence,
        compatibility=record.compatibility,
        d_tilde=d_norm,
        c_tilde=c_norm,
        d_l=d_norm * c_norm,
        d_i=d_norm * (1.0 - c_norm),
        s2=record.s2,
        repetition_rate=record.repetition_rate,
        p_hat=record.p_hat,
        alternative_direction_available=record.alternative_direction_available,
        batch_id=record.batch_id,
    )


def apply_frozen_scale(
    records: Sequence[RawStateSignal], scale: FrozenScaleArtifact
) -> List[StateSignal]:
    """Apply D1 anchors per state; batch_id is copied only as audit metadata."""

    output: List[StateSignal] = []
    for record in records:
        d_norm = _fixed_normalize(
            record.divergence,
            scale.divergence_q_low,
            scale.divergence_q_high,
        )
        c_norm = _fixed_normalize(
            record.compatibility,
            scale.compatibility_q_low,
            scale.compatibility_q_high,
        )
        output.append(_decompose_record(record, d_norm, c_norm))
    return output


def decompose_batch(
    records: Sequence[RawStateSignal], q_low: float = 0.05, q_high: float = 0.95
) -> List[StateSignal]:
    """Apply TA-OPD batch-wise robust normalization and D^L/D^I split."""

    if not records:
        return []
    d_tilde = robust_normalize([record.divergence for record in records], q_low, q_high)
    c_tilde = robust_normalize([record.compatibility for record in records], q_low, q_high)
    output: List[StateSignal] = []
    for record, d_norm, c_norm in zip(records, d_tilde, c_tilde):
        output.append(_decompose_record(record, d_norm, c_norm))
    return output


def decompose_grouped_batches(
    records: Sequence[RawStateSignal], q_low: float = 0.05, q_high: float = 0.95
) -> List[StateSignal]:
    """Normalize each recorded rollout batch independently while preserving row order."""

    groups: Dict[str, List[Tuple[int, RawStateSignal]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.batch_id].append((index, record))
    output: List[Optional[StateSignal]] = [None] * len(records)
    for batch_id in sorted(groups):
        indexed = groups[batch_id]
        decomposed = decompose_batch([record for _, record in indexed], q_low, q_high)
        for (index, _), state in zip(indexed, decomposed):
            output[index] = state
    if any(state is None for state in output):
        raise AssertionError("internal grouped decomposition error")
    return [state for state in output if state is not None]


def reflection_mass(teacher: Distribution, reflection_token_ids: Iterable[Token]) -> float:
    """Sum absolute untempered teacher probability on tokenizer-specific onset IDs."""

    _validate_distribution(teacher, "teacher")
    unique_ids = set(reflection_token_ids)
    if not unique_ids:
        raise ValueError("reflection_token_ids must not be empty")
    missing = unique_ids - set(teacher)
    if missing:
        raise ValueError(
            "teacher distribution is missing requested reflection token probabilities: "
            + ", ".join(sorted(str(token) for token in missing))
        )
    return sum(teacher[token_id] for token_id in unique_ids)


def handoff_trigger(
    teacher: Distribution,
    student: Distribution,
    reflection_token_ids: Iterable[Token],
    k: int = 5,
) -> bool:
    """Relay-style trigger: teacher top-1 reflects, student top-k does not."""

    _validate_distribution(teacher, "teacher")
    _validate_distribution(student, "student")
    reflection = set(reflection_token_ids)
    if not reflection:
        raise ValueError("reflection_token_ids must not be empty")
    if set(teacher) != set(student):
        raise ValueError(
            "teacher and student handoff distributions must use identical vocabulary keys"
        )
    missing_teacher = reflection - set(teacher)
    missing_student = reflection - set(student)
    if missing_teacher or missing_student:
        missing = sorted(str(token) for token in missing_teacher | missing_student)
        raise ValueError(
            "handoff distributions are missing reflection-token probabilities: "
            + ", ".join(missing)
        )
    teacher_top = _top_k(teacher, 1)[0]
    student_top = set(_top_k(student, k))
    return teacher_top in reflection and not student_top.intersection(reflection)
