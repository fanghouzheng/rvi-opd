from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Dict, Optional


def _require_nonempty_text(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _require_probability(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be a finite probability in [0, 1]")


class Action(str, Enum):
    REPAIR = "repair"
    INTERVENE = "intervene"
    DISCARD = "discard"
    DETACHED = "detached"


@dataclass(frozen=True)
class RawStateSignal:
    state_id: str
    problem_id: str
    trajectory_id: str
    token_index: int
    divergence: float
    compatibility: float
    s2: float
    repetition_rate: float = 0.0
    p_hat: float = 1.0
    alternative_direction_available: bool = False
    batch_id: str = "default"

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self, ("state_id", "problem_id", "trajectory_id", "batch_id")
        )
        if (
            isinstance(self.token_index, bool)
            or not isinstance(self.token_index, int)
            or self.token_index < 0
        ):
            raise ValueError("token_index must be a non-negative integer")
        if (
            isinstance(self.divergence, bool)
            or not isinstance(self.divergence, (int, float))
            or not math.isfinite(float(self.divergence))
            or self.divergence < 0
        ):
            raise ValueError("divergence must be finite and non-negative")
        _require_probability(
            self, ("compatibility", "s2", "repetition_rate", "p_hat")
        )
        if not isinstance(self.alternative_direction_available, bool):
            raise ValueError("alternative_direction_available must be boolean")


@dataclass(frozen=True)
class StateSignal:
    state_id: str
    problem_id: str
    trajectory_id: str
    token_index: int
    divergence: float
    compatibility: float
    d_tilde: float
    c_tilde: float
    d_l: float
    d_i: float
    s2: float
    repetition_rate: float = 0.0
    p_hat: float = 1.0
    alternative_direction_available: bool = False
    batch_id: str = "default"

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self, ("state_id", "problem_id", "trajectory_id", "batch_id")
        )
        if (
            isinstance(self.token_index, bool)
            or not isinstance(self.token_index, int)
            or self.token_index < 0
        ):
            raise ValueError("token_index must be a non-negative integer")
        if (
            isinstance(self.divergence, bool)
            or not isinstance(self.divergence, (int, float))
            or not math.isfinite(float(self.divergence))
            or self.divergence < 0
        ):
            raise ValueError("divergence must be finite and non-negative")
        _require_probability(
            self,
            (
                "compatibility",
                "d_tilde",
                "c_tilde",
                "d_l",
                "d_i",
                "s2",
                "repetition_rate",
                "p_hat",
            ),
        )
        if not math.isclose(self.d_l + self.d_i, self.d_tilde, abs_tol=1e-9):
            raise ValueError("d_l + d_i must equal d_tilde")
        if not isinstance(self.alternative_direction_available, bool):
            raise ValueError("alternative_direction_available must be boolean")

    @property
    def s1(self) -> float:
        return max(self.d_l, self.d_i)


@dataclass(frozen=True)
class RouteDecision:
    state_id: str
    problem_id: str
    trajectory_id: str
    token_index: int
    action: Action
    reason: str
    s1: float
    s2: float
    batch_id: str = "default"

    def __post_init__(self) -> None:
        _require_nonempty_text(
            self,
            ("state_id", "problem_id", "trajectory_id", "reason", "batch_id"),
        )
        if not isinstance(self.action, Action):
            raise ValueError("action must be an Action")
        if (
            isinstance(self.token_index, bool)
            or not isinstance(self.token_index, int)
            or self.token_index < 0
        ):
            raise ValueError("token_index must be a non-negative integer")
        _require_probability(self, ("s1", "s2"))


@dataclass(frozen=True)
class ActionAssignment:
    """Action plus its inseparable A2 payload/cost bundle."""

    state_id: str
    block_id: str
    action: Action
    bridge_token_length: int
    cost_signature: str
    payload_hash: str
    source_state_id: str = ""

    def __post_init__(self) -> None:
        if self.bridge_token_length < 0:
            raise ValueError("bridge_token_length cannot be negative")
        for name in ("state_id", "block_id", "cost_signature", "payload_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class CostVector:
    examples: int = 0
    teacher_scored_tokens: int = 0
    teacher_generated_tokens: int = 0
    teacher_inserted_tokens: int = 0
    student_supervised_tokens: int = 0
    teacher_forward_calls: int = 0
    gate_teacher_scored_tokens: int = 0
    teacher_prefill_tokens: int = 0
    student_rollout_tokens: int = 0
    optimizer_steps: int = 0
    teacher_gpu_seconds: float = 0.0
    student_gpu_seconds: float = 0.0
    wall_time_ms: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "examples",
            "teacher_scored_tokens",
            "teacher_generated_tokens",
            "teacher_inserted_tokens",
            "student_supervised_tokens",
            "teacher_forward_calls",
            "gate_teacher_scored_tokens",
            "teacher_prefill_tokens",
            "student_rollout_tokens",
            "optimizer_steps",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{field.name} must be a non-negative integer")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a non-negative finite number")
            elif not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{field.name} must be a non-negative finite number")

    def __add__(self, other: "CostVector") -> "CostVector":
        if not isinstance(other, CostVector):
            return NotImplemented
        return CostVector(
            examples=self.examples + other.examples,
            teacher_scored_tokens=self.teacher_scored_tokens + other.teacher_scored_tokens,
            teacher_generated_tokens=self.teacher_generated_tokens + other.teacher_generated_tokens,
            teacher_inserted_tokens=self.teacher_inserted_tokens + other.teacher_inserted_tokens,
            student_supervised_tokens=self.student_supervised_tokens
            + other.student_supervised_tokens,
            teacher_forward_calls=self.teacher_forward_calls + other.teacher_forward_calls,
            gate_teacher_scored_tokens=self.gate_teacher_scored_tokens
            + other.gate_teacher_scored_tokens,
            teacher_prefill_tokens=self.teacher_prefill_tokens + other.teacher_prefill_tokens,
            student_rollout_tokens=self.student_rollout_tokens + other.student_rollout_tokens,
            optimizer_steps=self.optimizer_steps + other.optimizer_steps,
            teacher_gpu_seconds=self.teacher_gpu_seconds + other.teacher_gpu_seconds,
            student_gpu_seconds=self.student_gpu_seconds + other.student_gpu_seconds,
            wall_time_ms=self.wall_time_ms + other.wall_time_ms,
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    requested_action: Action
    effective_action: Action
    accepted: bool
    s2_before: float
    s2_after: float
    s2_drop: float
    teacher_preferred_before: float
    teacher_preferred_after: float
    teacher_preferred_gain: float
    reason: str
    gate_mode: str = "absolute_smoke"
    gate_statistic: Optional[float] = None
    gate_threshold: Optional[float] = None
    gate_artifact_sha256: str = ""


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int
    resamples: int
    seed: int


@dataclass(frozen=True)
class BudgetMismatch:
    field: str
    left: float
    right: float
    absolute_difference: float
    tolerance: float
    note: Optional[str] = None
