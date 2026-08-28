from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from statistics import mean, pstdev
from typing import Dict, Mapping, Sequence, Tuple

from .models import Action, GateDecision
from .signals import quantile


@dataclass(frozen=True)
class GateConfig:
    """Absolute synthetic-smoke thresholds; not the production acceptance gate."""

    min_rollouts: int = 4
    min_s2_drop: float = 0.001
    min_teacher_preferred_gain: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.min_rollouts, bool) or not isinstance(self.min_rollouts, int):
            raise ValueError("min_rollouts must be a positive integer")
        if self.min_rollouts <= 0:
            raise ValueError("min_rollouts must be a positive integer")
        for name in ("min_s2_drop", "min_teacher_preferred_gain"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite and non-negative")
            if value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class FrozenJointGateArtifact:
    """D1 null calibration for one event-wise joint max-statistic gate."""

    s2_drop_null_mean: float
    s2_drop_null_std: float
    agreement_gain_null_mean: float
    agreement_gain_null_std: float
    max_stat_q95: float
    record_count: int
    null_quantile: float = 0.95
    calibration_split_sha256: str = ""
    threshold_artifact_sha256: str = ""
    code_revision: str = ""
    schema_version: str = "rvi-joint-gate-v1"

    def __post_init__(self) -> None:
        for name in (
            "s2_drop_null_mean",
            "s2_drop_null_std",
            "agreement_gain_null_mean",
            "agreement_gain_null_std",
            "max_stat_q95",
            "null_quantile",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.s2_drop_null_std <= 0 or self.agreement_gain_null_std <= 0:
            raise ValueError("joint-gate null standard deviations must be positive")
        if not 0 < self.null_quantile < 1:
            raise ValueError("joint-gate null quantile must be in (0, 1)")
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 2
        ):
            raise ValueError("joint-gate calibration needs at least two null records")
        if self.schema_version != "rvi-joint-gate-v1":
            raise ValueError("unsupported joint-gate schema version")

    def _unsigned_payload(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def artifact_sha256(self) -> str:
        encoded = json.dumps(
            self._unsigned_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def production_ready(self) -> bool:
        return bool(
            self.null_quantile == 0.95
            and re.fullmatch(r"[0-9a-f]{64}", self.calibration_split_sha256)
            and re.fullmatch(r"[0-9a-f]{64}", self.threshold_artifact_sha256)
            and re.fullmatch(r"[0-9a-f]{40}", self.code_revision)
        )

    def assert_production_ready(self) -> None:
        if not self.production_ready:
            raise ValueError(
                "joint-gate artifact needs calibration/threshold SHA256 hashes and a "
                "40-hex code revision"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            **self._unsigned_payload(),
            "production_ready": self.production_ready,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenJointGateArtifact":
        expected_hash = payload.get("artifact_sha256")
        field_names = {field.name for field in fields(cls)}
        unknown = set(payload) - field_names - {"production_ready", "artifact_sha256"}
        if unknown:
            raise ValueError("unknown joint-gate artifact fields: " + ", ".join(sorted(unknown)))
        try:
            artifact = cls(**{key: payload[key] for key in field_names if key in payload})
        except TypeError as exc:
            raise ValueError(f"invalid joint-gate artifact: {exc}") from exc
        if expected_hash is not None and expected_hash != artifact.artifact_sha256:
            raise ValueError("joint-gate artifact SHA256 does not match its contents")
        return artifact


def fit_joint_gate_artifact(
    null_s2_drops: Sequence[float],
    null_agreement_gains: Sequence[float],
    quantile_level: float = 0.95,
    calibration_split_sha256: str = "",
    threshold_artifact_sha256: str = "",
    code_revision: str = "",
) -> FrozenJointGateArtifact:
    """Fit a joint max-statistic null from paired ineffective/random bridges."""

    if len(null_s2_drops) != len(null_agreement_gains) or len(null_s2_drops) < 2:
        raise ValueError("joint-gate null arrays must be paired and contain at least two rows")
    if not 0 < quantile_level < 1:
        raise ValueError("joint-gate quantile must be in (0, 1)")
    values = [*null_s2_drops, *null_agreement_gains]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -1 <= value <= 1
        for value in values
    ):
        raise ValueError("joint-gate null improvements must be finite and in [-1, 1]")
    s2_center = mean(null_s2_drops)
    agreement_center = mean(null_agreement_gains)
    s2_scale = pstdev(null_s2_drops)
    agreement_scale = pstdev(null_agreement_gains)
    if s2_scale <= 0 or agreement_scale <= 0:
        raise ValueError("joint-gate null improvements need non-zero variance on both metrics")
    max_statistics = [
        max(
            (s2_drop - s2_center) / s2_scale,
            (agreement_gain - agreement_center) / agreement_scale,
        )
        for s2_drop, agreement_gain in zip(null_s2_drops, null_agreement_gains)
    ]
    return FrozenJointGateArtifact(
        s2_drop_null_mean=s2_center,
        s2_drop_null_std=s2_scale,
        agreement_gain_null_mean=agreement_center,
        agreement_gain_null_std=agreement_scale,
        max_stat_q95=quantile(max_statistics, quantile_level),
        record_count=len(null_s2_drops),
        null_quantile=quantile_level,
        calibration_split_sha256=calibration_split_sha256,
        threshold_artifact_sha256=threshold_artifact_sha256,
        code_revision=code_revision,
    )


def _summarize_paired_probes(
    s2_before: Sequence[float],
    s2_after: Sequence[float],
    agreement_before: Sequence[float],
    agreement_after: Sequence[float],
    min_rollouts: int,
) -> Tuple[float, float, float, float, float, float]:
    lengths = {
        len(s2_before),
        len(s2_after),
        len(agreement_before),
        len(agreement_after),
    }
    if len(lengths) != 1:
        raise ValueError("gate probes must be paired and have equal lengths")
    sample_count = next(iter(lengths))
    if sample_count < min_rollouts:
        raise ValueError(f"gate needs at least {min_rollouts} paired rollouts; got {sample_count}")
    values = [*s2_before, *s2_after, *agreement_before, *agreement_after]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("gate probe metrics must be finite probabilities in [0, 1]")
    before_s2 = mean(s2_before)
    after_s2 = mean(s2_after)
    before_agreement = mean(agreement_before)
    after_agreement = mean(agreement_after)
    return (
        before_s2,
        after_s2,
        before_s2 - after_s2,
        before_agreement,
        after_agreement,
        after_agreement - before_agreement,
    )


def evaluate_joint_intervention_gate(
    s2_before: Sequence[float],
    s2_after: Sequence[float],
    agreement_before: Sequence[float],
    agreement_after: Sequence[float],
    artifact: FrozenJointGateArtifact,
    min_rollouts: int = 4,
) -> GateDecision:
    """Production gate: one test against the frozen joint max-statistic q95."""

    if isinstance(min_rollouts, bool) or not isinstance(min_rollouts, int) or min_rollouts <= 0:
        raise ValueError("min_rollouts must be a positive integer")
    (
        before_s2,
        after_s2,
        s2_drop,
        before_agreement,
        after_agreement,
        agreement_gain,
    ) = _summarize_paired_probes(
        s2_before,
        s2_after,
        agreement_before,
        agreement_after,
        min_rollouts,
    )
    statistic = max(
        (s2_drop - artifact.s2_drop_null_mean) / artifact.s2_drop_null_std,
        (agreement_gain - artifact.agreement_gain_null_mean)
        / artifact.agreement_gain_null_std,
    )
    accepted = statistic > artifact.max_stat_q95
    return GateDecision(
        requested_action=Action.INTERVENE,
        effective_action=Action.INTERVENE if accepted else Action.REPAIR,
        accepted=accepted,
        s2_before=before_s2,
        s2_after=after_s2,
        s2_drop=s2_drop,
        teacher_preferred_before=before_agreement,
        teacher_preferred_after=after_agreement,
        teacher_preferred_gain=agreement_gain,
        reason=(
            "accepted_joint_max_statistic"
            if accepted
            else "rollback_joint_max_statistic_below_frozen_q95"
        ),
        gate_mode="joint_max_statistic",
        gate_statistic=statistic,
        gate_threshold=artifact.max_stat_q95,
        gate_artifact_sha256=artifact.artifact_sha256,
    )


def evaluate_intervention_gate(
    s2_before: Sequence[float],
    s2_after: Sequence[float],
    teacher_preferred_before: Sequence[float],
    teacher_preferred_after: Sequence[float],
    config: GateConfig = GateConfig(),
) -> GateDecision:
    (
        before_s2,
        after_s2,
        s2_drop,
        before_preferred,
        after_preferred,
        preferred_gain,
    ) = _summarize_paired_probes(
        s2_before,
        s2_after,
        teacher_preferred_before,
        teacher_preferred_after,
        config.min_rollouts,
    )
    accepted = (
        s2_drop >= config.min_s2_drop
        or preferred_gain >= config.min_teacher_preferred_gain
    )
    if s2_drop >= config.min_s2_drop:
        reason = "accepted_s2_residual_drop"
    elif preferred_gain >= config.min_teacher_preferred_gain:
        reason = "accepted_teacher_preferred_gain"
    else:
        reason = "rollback_no_paired_probe_improvement"

    return GateDecision(
        requested_action=Action.INTERVENE,
        effective_action=Action.INTERVENE if accepted else Action.REPAIR,
        accepted=accepted,
        s2_before=before_s2,
        s2_after=after_s2,
        s2_drop=s2_drop,
        teacher_preferred_before=before_preferred,
        teacher_preferred_after=after_preferred,
        teacher_preferred_gain=preferred_gain,
        reason=reason,
        gate_mode="absolute_smoke",
        gate_statistic=max(s2_drop, preferred_gain),
        gate_threshold=None,
    )
