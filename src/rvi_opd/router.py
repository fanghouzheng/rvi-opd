from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import Action, ActionAssignment, RouteDecision, StateSignal
from .signals import FrozenScaleArtifact, quantile


PRIMARY_ROUTER_QUANTILE = 0.80


@dataclass(frozen=True)
class ThresholdArtifact:
    s1_threshold: float
    s2_threshold: float
    s1_quantile: float
    s2_quantile: float
    calibration_fingerprint: str
    record_count: int
    signal_schema_version: str = "rvi-signals-v3"
    teacher_revision: str = ""
    student_revision: str = ""
    tokenizer_sha256: str = ""
    vocabulary_sha256: str = ""
    code_revision: str = ""
    calibration_split_sha256: str = ""
    normalization_low: float = 0.05
    normalization_high: float = 0.95
    divergence_q_low: Optional[float] = None
    divergence_q_high: Optional[float] = None
    compatibility_q_low: Optional[float] = None
    compatibility_q_high: Optional[float] = None
    scale_artifact_sha256: str = ""
    # Legacy single-lexicon fields remain readable but can never satisfy the
    # production contract, which requires distinct TRD and Relay artifacts.
    lexicon_artifact_sha256: str = ""
    reflection_token_ids: Tuple[int, ...] = ()
    trd_epistemic_lexicon_artifact_sha256: str = ""
    relay_single_token_lexicon_artifact_sha256: str = ""
    trd_epistemic_token_ids: Tuple[int, ...] = ()
    relay_single_token_ids: Tuple[int, ...] = ()
    s1_q25: Optional[float] = None
    s1_q75: Optional[float] = None
    s2_q25: Optional[float] = None
    s2_q75: Optional[float] = None
    threshold_scope: str = "global_frozen"

    def __post_init__(self) -> None:
        numeric = {
            "s1_threshold": self.s1_threshold,
            "s2_threshold": self.s2_threshold,
            "s1_quantile": self.s1_quantile,
            "s2_quantile": self.s2_quantile,
            "normalization_low": self.normalization_low,
            "normalization_high": self.normalization_high,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric.values()
        ):
            raise ValueError("threshold artifact numeric fields must be finite")
        if not 0 <= self.s1_threshold <= 1 or not 0 <= self.s2_threshold <= 1:
            raise ValueError("signal thresholds must be in [0, 1]")
        if not 0 <= self.s1_quantile <= 1 or not 0 <= self.s2_quantile <= 1:
            raise ValueError("threshold quantiles must be in [0, 1]")
        if not 0 <= self.normalization_low < self.normalization_high <= 1:
            raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1")
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int):
            raise ValueError("record_count must be an integer")
        if self.record_count <= 0:
            raise ValueError("record_count must be positive")
        text_names = (
            "calibration_fingerprint",
            "signal_schema_version",
            "teacher_revision",
            "student_revision",
            "tokenizer_sha256",
            "vocabulary_sha256",
            "code_revision",
            "calibration_split_sha256",
            "scale_artifact_sha256",
            "lexicon_artifact_sha256",
            "trd_epistemic_lexicon_artifact_sha256",
            "relay_single_token_lexicon_artifact_sha256",
        )
        if any(not isinstance(getattr(self, name), str) for name in text_names):
            raise ValueError("threshold artifact metadata fields must be strings")
        anchor_names = (
            "divergence_q_low",
            "divergence_q_high",
            "compatibility_q_low",
            "compatibility_q_high",
        )
        anchors = [getattr(self, name) for name in anchor_names]
        if any(value is not None for value in anchors) and not all(
            value is not None for value in anchors
        ):
            raise ValueError("threshold artifact must contain all four frozen-scale anchors")
        if all(value is not None for value in anchors):
            for name in anchor_names:
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{name} must be a finite number")
                object.__setattr__(self, name, float(value))
            self.frozen_scale()
        elif self.scale_artifact_sha256:
            raise ValueError("frozen-scale SHA256 requires all four scale anchors")
        for name in (
            "reflection_token_ids",
            "trd_epistemic_token_ids",
            "relay_single_token_ids",
        ):
            token_ids = getattr(self, name)
            if isinstance(token_ids, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of token IDs")
            normalized_ids = []
            for token_id in token_ids:
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    raise ValueError(f"{name} must contain non-negative integer token IDs")
                normalized_ids.append(token_id)
            object.__setattr__(self, name, tuple(sorted(set(normalized_ids))))
        band_values = (self.s1_q25, self.s1_q75, self.s2_q25, self.s2_q75)
        if any(value is not None for value in band_values):
            if not all(
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and 0 <= float(value) <= 1
                for value in band_values
            ):
                raise ValueError("all s1/s2 q25/q75 band thresholds must be finite probabilities")
            if self.s1_q25 > self.s1_q75 or self.s2_q25 > self.s2_q75:
                raise ValueError("s1/s2 q25 thresholds cannot exceed q75 thresholds")
            for name in ("s1_q25", "s1_q75", "s2_q25", "s2_q75"):
                object.__setattr__(self, name, float(getattr(self, name)))
        if self.threshold_scope != "global_frozen":
            raise ValueError("threshold_scope must be global_frozen")

    def to_dict(self) -> Dict[str, object]:
        return {
            **self._unsigned_payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def _unsigned_payload(self) -> Dict[str, object]:
        return {**asdict(self), "production_ready": self.production_ready}

    @property
    def artifact_sha256(self) -> str:
        encoded = json.dumps(
            self._unsigned_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ThresholdArtifact":
        expected_hash = payload.get("artifact_sha256")
        field_names = {field.name for field in fields(cls)}
        unknown = set(payload) - field_names - {"production_ready", "artifact_sha256"}
        if unknown:
            raise ValueError("unknown threshold artifact fields: " + ", ".join(sorted(unknown)))
        try:
            artifact = cls(**{key: payload[key] for key in field_names if key in payload})
        except TypeError as exc:
            raise ValueError(f"invalid threshold artifact: {exc}") from exc
        if expected_hash is not None and expected_hash != artifact.artifact_sha256:
            raise ValueError("threshold artifact SHA256 does not match its contents")
        return artifact

    @property
    def production_ready(self) -> bool:
        try:
            scale = self.frozen_scale()
        except ValueError:
            return False
        band_values_ready = bool(
            self.s1_q25 is not None
            and self.s1_q75 is not None
            and self.s2_q25 is not None
            and self.s2_q75 is not None
            and self.s1_q25 < self.s1_q75
            and self.s2_q25 < self.s2_q75
        )
        return bool(
            self.signal_schema_version == "rvi-signals-v3"
            and self.normalization_low == 0.05
            and self.normalization_high == 0.95
            and self.s1_quantile == PRIMARY_ROUTER_QUANTILE
            and self.s2_quantile == PRIMARY_ROUTER_QUANTILE
            and re.fullmatch(r"[0-9a-f]{64}", self.calibration_fingerprint)
            and re.fullmatch(r"[0-9a-f]{40}", self.teacher_revision)
            and re.fullmatch(r"[0-9a-f]{40}", self.student_revision)
            and re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_sha256)
            and scale.production_ready
            and re.fullmatch(r"[0-9a-f]{64}", self.calibration_split_sha256)
            and re.fullmatch(r"[0-9a-f]{64}", self.scale_artifact_sha256)
            and self.scale_artifact_sha256 == scale.artifact_sha256
            and re.fullmatch(
                r"[0-9a-f]{64}", self.trd_epistemic_lexicon_artifact_sha256
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", self.relay_single_token_lexicon_artifact_sha256
            )
            and bool(self.trd_epistemic_token_ids)
            and bool(self.relay_single_token_ids)
            and band_values_ready
            and self.s1_threshold >= self.s1_q75
            and self.s2_threshold >= self.s2_q75
        )

    def frozen_scale(self) -> FrozenScaleArtifact:
        anchors = (
            self.divergence_q_low,
            self.divergence_q_high,
            self.compatibility_q_low,
            self.compatibility_q_high,
        )
        if any(value is None for value in anchors):
            raise ValueError("threshold artifact is missing frozen raw-D/raw-C anchors")
        scale = FrozenScaleArtifact(
            q_low=self.normalization_low,
            q_high=self.normalization_high,
            divergence_q_low=self.divergence_q_low,
            divergence_q_high=self.divergence_q_high,
            compatibility_q_low=self.compatibility_q_low,
            compatibility_q_high=self.compatibility_q_high,
            vocabulary_sha256=self.vocabulary_sha256,
            code_revision=self.code_revision,
        )
        if not self.scale_artifact_sha256:
            raise ValueError("threshold artifact is missing its frozen-scale SHA256")
        if self.scale_artifact_sha256 != scale.artifact_sha256:
            raise ValueError("frozen-scale SHA256 does not match threshold anchors")
        return scale

    def assert_production_ready(self) -> None:
        if not self.production_ready:
            raise ValueError(
                "threshold artifact needs schema v3, q05/q95 normalization, primary q80 "
                "thresholds consistent with global q25/q75 bands, frozen raw-D/raw-C anchors "
                "and scale hash, a calibration fingerprint, "
                "40-hex model/code revisions, 64-hex tokenizer/vocabulary/split and distinct "
                "TRD/Relay lexicon hashes, plus both non-empty token-ID sets"
            )


@dataclass(frozen=True)
class RouterPolicy:
    """Optional boundary-condition routes; both are disabled in the core confirmatory router."""

    repetition_threshold: Optional[float] = None
    paced_zero_rescue: bool = False

    def __post_init__(self) -> None:
        if self.repetition_threshold is not None and (
            isinstance(self.repetition_threshold, bool)
            or not isinstance(self.repetition_threshold, (int, float))
            or not math.isfinite(float(self.repetition_threshold))
            or not 0 <= self.repetition_threshold <= 1
        ):
            raise ValueError("repetition_threshold must be a finite probability in [0, 1]")
        if not isinstance(self.paced_zero_rescue, bool):
            raise ValueError("paced_zero_rescue must be boolean")


def _fingerprint(records: Sequence[StateSignal]) -> str:
    payload = [
        {
            "state_id": record.state_id,
            "batch_id": record.batch_id,
            "s1": record.s1,
            "s2": record.s2,
        }
        for record in sorted(records, key=lambda item: item.state_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibrate_thresholds(
    calibration_records: Sequence[StateSignal],
    s1_quantile: float = PRIMARY_ROUTER_QUANTILE,
    s2_quantile: float = PRIMARY_ROUTER_QUANTILE,
    teacher_revision: str = "",
    student_revision: str = "",
    tokenizer_sha256: str = "",
    calibration_split_sha256: str = "",
    normalization_low: Optional[float] = None,
    normalization_high: Optional[float] = None,
    lexicon_artifact_sha256: str = "",
    reflection_token_ids: Sequence[int] = (),
    frozen_scale: Optional[FrozenScaleArtifact] = None,
    trd_epistemic_lexicon_artifact_sha256: str = "",
    relay_single_token_lexicon_artifact_sha256: str = "",
    trd_epistemic_token_ids: Sequence[int] = (),
    relay_single_token_ids: Sequence[int] = (),
) -> ThresholdArtifact:
    if not calibration_records:
        raise ValueError("calibration_records must not be empty")
    state_ids = [record.state_id for record in calibration_records]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("calibration_records must contain unique state IDs")
    if frozen_scale is None:
        normalization_low = 0.05 if normalization_low is None else normalization_low
        normalization_high = 0.95 if normalization_high is None else normalization_high
        divergence_q_low = None
        divergence_q_high = None
        compatibility_q_low = None
        compatibility_q_high = None
        scale_artifact_sha256 = ""
        vocabulary_sha256 = ""
        code_revision = ""
    else:
        if normalization_low is not None and normalization_low != frozen_scale.q_low:
            raise ValueError("normalization_low does not match the frozen-scale artifact")
        if normalization_high is not None and normalization_high != frozen_scale.q_high:
            raise ValueError("normalization_high does not match the frozen-scale artifact")
        normalization_low = frozen_scale.q_low
        normalization_high = frozen_scale.q_high
        divergence_q_low = frozen_scale.divergence_q_low
        divergence_q_high = frozen_scale.divergence_q_high
        compatibility_q_low = frozen_scale.compatibility_q_low
        compatibility_q_high = frozen_scale.compatibility_q_high
        scale_artifact_sha256 = frozen_scale.artifact_sha256
        vocabulary_sha256 = frozen_scale.vocabulary_sha256
        code_revision = frozen_scale.code_revision
    return ThresholdArtifact(
        s1_threshold=quantile([record.s1 for record in calibration_records], s1_quantile),
        s2_threshold=quantile([record.s2 for record in calibration_records], s2_quantile),
        s1_quantile=s1_quantile,
        s2_quantile=s2_quantile,
        calibration_fingerprint=_fingerprint(calibration_records),
        record_count=len(calibration_records),
        teacher_revision=teacher_revision,
        student_revision=student_revision,
        tokenizer_sha256=tokenizer_sha256,
        vocabulary_sha256=vocabulary_sha256,
        code_revision=code_revision,
        calibration_split_sha256=calibration_split_sha256,
        normalization_low=normalization_low,
        normalization_high=normalization_high,
        divergence_q_low=divergence_q_low,
        divergence_q_high=divergence_q_high,
        compatibility_q_low=compatibility_q_low,
        compatibility_q_high=compatibility_q_high,
        scale_artifact_sha256=scale_artifact_sha256,
        lexicon_artifact_sha256=lexicon_artifact_sha256,
        reflection_token_ids=tuple(reflection_token_ids),
        trd_epistemic_lexicon_artifact_sha256=trd_epistemic_lexicon_artifact_sha256,
        relay_single_token_lexicon_artifact_sha256=relay_single_token_lexicon_artifact_sha256,
        trd_epistemic_token_ids=tuple(trd_epistemic_token_ids),
        relay_single_token_ids=tuple(relay_single_token_ids),
        s1_q25=quantile([record.s1 for record in calibration_records], 0.25),
        s1_q75=quantile([record.s1 for record in calibration_records], 0.75),
        s2_q25=quantile([record.s2 for record in calibration_records], 0.25),
        s2_q75=quantile([record.s2 for record in calibration_records], 0.75),
    )


def route_state(
    record: StateSignal,
    thresholds: ThresholdArtifact,
    policy: RouterPolicy = RouterPolicy(),
) -> RouteDecision:
    s2_stratum = record.s2_analysis_stratum
    if s2_stratum == "unknown" and thresholds.s2_q25 is not None and thresholds.s2_q75 is not None:
        if record.s2 <= thresholds.s2_q25:
            s2_stratum = "low"
        elif record.s2 >= thresholds.s2_q75:
            s2_stratum = "high"
        else:
            s2_stratum = "middle"
    intervention_available = (
        record.relay_phi_eligible
        and record.intervention_budget_available
        and record.intervention_cooldown_available
    )
    if (
        policy.repetition_threshold is not None
        and record.repetition_rate >= policy.repetition_threshold
    ):
        if record.alternative_direction_available and record.intervention_budget_available:
            action = Action.INTERVENE
            reason = "d4_repetition_bypass_with_teacher_alternative_budget_eligible_nonrelay_phi"
        else:
            action = Action.DISCARD
            reason = (
                "d4_repetition_bypass_without_budget_or_teacher_alternative"
                if not record.intervention_budget_available
                else "d4_repetition_bypass_without_teacher_alternative"
            )
    elif policy.paced_zero_rescue and record.p_hat <= 0 and intervention_available:
        action = Action.INTERVENE
        reason = "d5_zero_pass_rate_rescue_phi_budget_cooldown_eligible"
    elif record.s2 >= thresholds.s2_threshold and intervention_available:
        action = Action.INTERVENE
        reason = "high_s2_relay_phi_budget_cooldown_eligible"
    elif record.s1 >= thresholds.s1_threshold:
        action = Action.REPAIR
        reason = (
            "d5_zero_pass_rescue_intervention_unavailable_high_s1_repair"
            if policy.paced_zero_rescue and record.p_hat <= 0
            else "high_s2_intervention_unavailable_high_s1_repair"
            if record.s2 >= thresholds.s2_threshold
            else "high_s1_low_s2_locally_absorbable"
        )
    else:
        action = Action.DISCARD
        reason = (
            "d5_zero_pass_rescue_intervention_unavailable_low_s1_discard"
            if policy.paced_zero_rescue and record.p_hat <= 0
            else "high_s2_intervention_unavailable_low_s1_discard"
            if record.s2 >= thresholds.s2_threshold
            else "low_s1_low_s2"
        )
    return RouteDecision(
        state_id=record.state_id,
        problem_id=record.problem_id,
        trajectory_id=record.trajectory_id,
        token_index=record.token_index,
        action=action,
        reason=reason,
        s1=record.s1,
        s2=record.s2,
        batch_id=record.batch_id,
        max_response_tokens=record.max_response_tokens,
        difficulty_bin=record.difficulty_bin,
        prefix_position_stratum=record.prefix_position_stratum,
        s2_analysis_stratum=s2_stratum,
        trigger_manifest_sha256=record.trigger_manifest_sha256,
        relay_phi_eligible=record.relay_phi_eligible,
        intervention_budget_available=record.intervention_budget_available,
        intervention_cooldown_available=record.intervention_cooldown_available,
    )


def route_batch(
    records: Iterable[StateSignal],
    thresholds: ThresholdArtifact,
    policy: RouterPolicy = RouterPolicy(),
) -> List[RouteDecision]:
    return [route_state(record, thresholds, policy) for record in records]


def permute_actions_within_blocks(
    decisions: Sequence[RouteDecision], seed: int, block_by: str = "problem_id"
) -> List[RouteDecision]:
    """Deprecated unsafe A2 helper; label-only shuffling is prohibited."""

    raise RuntimeError(
        "label-only action permutation is not a valid A2 control; use "
        "permute_action_bundles_within_blocks with requested/effective actions, "
        "gate metadata, payload, realized length, and cost bound together"
    )


def _sattolo_indices(size: int, rng: random.Random) -> List[int]:
    indices = list(range(size))
    for index in range(size - 1, 0, -1):
        other = rng.randrange(index)
        indices[index], indices[other] = indices[other], indices[index]
    return indices


def permute_action_bundles_within_blocks(
    assignments: Sequence[ActionAssignment], seed: int
) -> List[ActionAssignment]:
    """A2 production helper: derange one complete post-gate action bundle."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not assignments:
        raise ValueError("A2 bundle permutation requires at least one non-empty block")
    blocks: Dict[str, List[ActionAssignment]] = {}
    all_state_ids = [assignment.state_id for assignment in assignments]
    if len(all_state_ids) != len(set(all_state_ids)):
        raise ValueError("A2 assignment state IDs must be globally unique")
    for assignment in assignments:
        blocks.setdefault(assignment.block_id, []).append(assignment)
    for block_id, block in blocks.items():
        state_ids = [assignment.state_id for assignment in block]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError(f"A2 block {block_id!r} contains duplicate state IDs")
        if len(block) < 2:
            raise ValueError(f"A2 block {block_id!r} must contain at least two states")
    rng = random.Random(seed)
    output: List[ActionAssignment] = []
    for block_id in sorted(blocks):
        recipients = blocks[block_id]
        donor_indices = _sattolo_indices(len(recipients), rng)
        for recipient, donor_index in zip(recipients, donor_indices):
            donor = recipients[donor_index]
            output.append(
                ActionAssignment(
                    state_id=recipient.state_id,
                    block_id=recipient.block_id,
                    requested_action=donor.requested_action,
                    effective_action=donor.effective_action,
                    bridge_token_length=donor.bridge_token_length,
                    cost_signature=donor.cost_signature,
                    payload_hash=donor.payload_hash,
                    gate_status=donor.gate_status,
                    gate_artifact_sha256=donor.gate_artifact_sha256,
                    source_state_id=donor.source_state_id,
                )
            )
    return output


def action_counts(decisions: Iterable[RouteDecision]) -> Mapping[str, int]:
    counts = {action.value: 0 for action in Action}
    for decision in decisions:
        counts[decision.action.value] += 1
    return counts
