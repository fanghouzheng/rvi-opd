import unittest

from rvi_opd.models import Action, ActionAssignment, StateSignal
from rvi_opd.router import (
    ThresholdArtifact,
    RouterPolicy,
    calibrate_thresholds,
    permute_actions_within_blocks,
    permute_action_bundles_within_blocks,
    route_batch,
    route_state,
)
from rvi_opd.signals import FrozenScaleArtifact


def state(
    state_id: str,
    problem_id: str = "p",
    d_l: float = 0.0,
    d_i: float = 0.0,
    s2: float = 0.0,
    repetition_rate: float = 0.0,
    p_hat: float = 1.0,
    alternative_direction_available: bool = False,
    relay_phi_eligible: bool = False,
    intervention_budget_available: bool = False,
    intervention_cooldown_available: bool = False,
) -> StateSignal:
    return StateSignal(
        state_id=state_id,
        problem_id=problem_id,
        trajectory_id="t-" + state_id,
        token_index=0,
        divergence=d_l + d_i,
        compatibility=0.5,
        d_tilde=d_l + d_i,
        c_tilde=0.5,
        d_l=d_l,
        d_i=d_i,
        s2=s2,
        repetition_rate=repetition_rate,
        p_hat=p_hat,
        alternative_direction_available=alternative_direction_available,
        relay_phi_eligible=relay_phi_eligible,
        intervention_budget_available=intervention_budget_available,
        intervention_cooldown_available=intervention_cooldown_available,
    )


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ThresholdArtifact(0.5, 0.01, 0.8, 0.8, "fixture", 10)

    def test_high_s2_requires_phi_budget_and_cooldown(self) -> None:
        decision = route_state(
            state(
                "x",
                d_l=0.9,
                s2=0.02,
                relay_phi_eligible=True,
                intervention_budget_available=True,
                intervention_cooldown_available=True,
            ),
            self.thresholds,
        )
        self.assertEqual(decision.action, Action.INTERVENE)
        self.assertTrue(decision.relay_phi_eligible)

        for missing in (
            "relay_phi_eligible",
            "intervention_budget_available",
            "intervention_cooldown_available",
        ):
            kwargs = {
                "relay_phi_eligible": True,
                "intervention_budget_available": True,
                "intervention_cooldown_available": True,
            }
            kwargs[missing] = False
            fallback = route_state(
                state("fallback-" + missing, d_l=0.9, s2=0.02, **kwargs),
                self.thresholds,
            )
            self.assertEqual(fallback.action, Action.REPAIR)
            self.assertIn("intervention_unavailable", fallback.reason)

        discarded = route_state(state("unavailable", s2=0.02), self.thresholds)
        self.assertEqual(discarded.action, Action.DISCARD)
        self.assertIn("intervention_unavailable", discarded.reason)

    def test_repair_and_discard(self) -> None:
        self.assertEqual(
            route_state(state("repair", d_i=0.8, s2=0.001), self.thresholds).action,
            Action.REPAIR,
        )
        self.assertEqual(
            route_state(state("discard", d_i=0.1, s2=0.001), self.thresholds).action,
            Action.DISCARD,
        )

    def test_calibration_fingerprint_is_order_independent(self) -> None:
        records = [state("a", d_l=0.2), state("b", d_i=0.9, s2=0.02)]
        first = calibrate_thresholds(records)
        second = calibrate_thresholds(list(reversed(records)))
        self.assertEqual(first.calibration_fingerprint, second.calibration_fingerprint)

    def test_calibration_rejects_duplicate_state_ids(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_thresholds([state("duplicate"), state("duplicate", d_l=0.8)])

    def test_thresholds_and_policy_are_probability_bounded(self) -> None:
        for s1_threshold, s2_threshold in (
            (-0.1, 0.1),
            (0.1, 1.1),
            (True, 0.1),
        ):
            with self.assertRaises(ValueError):
                ThresholdArtifact(
                    s1_threshold,
                    s2_threshold,
                    0.8,
                    0.8,
                    "fixture",
                    1,
                )
        for repetition_threshold in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                RouterPolicy(repetition_threshold=repetition_threshold)
        with self.assertRaises(ValueError):
            RouterPolicy(paced_zero_rescue=1)

    def test_production_threshold_metadata_is_enforced(self) -> None:
        artifact = calibrate_thresholds([state("a")])
        self.assertFalse(artifact.production_ready)
        with self.assertRaises(ValueError):
            artifact.assert_production_ready()
        scale = FrozenScaleArtifact(
            0.05,
            0.95,
            0.0,
            1.0,
            0.0,
            1.0,
            vocabulary_sha256="f" * 64,
            code_revision="1" * 40,
        )
        ready = calibrate_thresholds(
            [state("a"), state("b", d_l=0.8, s2=0.8)],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            trd_epistemic_lexicon_artifact_sha256="d" * 64,
            relay_single_token_lexicon_artifact_sha256="e" * 64,
            trd_epistemic_token_ids=[10, 20],
            relay_single_token_ids=[30, 40],
            frozen_scale=scale,
        )
        ready.assert_production_ready()
        self.assertEqual(ready.trd_epistemic_token_ids, (10, 20))
        self.assertEqual(ready.relay_single_token_ids, (30, 40))
        self.assertEqual(ready.scale_artifact_sha256, scale.artifact_sha256)
        self.assertEqual(ready.vocabulary_sha256, "f" * 64)
        self.assertEqual(ready.code_revision, "1" * 40)

        legacy_only = calibrate_thresholds(
            [state("legacy-a"), state("legacy-b", d_l=0.8, s2=0.8)],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            lexicon_artifact_sha256="e" * 64,
            reflection_token_ids=[10, 20],
            frozen_scale=scale,
        )
        self.assertFalse(legacy_only.production_ready)

        wrong_primary_quantile = calibrate_thresholds(
            [state("q-a"), state("q-b", d_l=0.8, s2=0.8)],
            s1_quantile=0.75,
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            trd_epistemic_lexicon_artifact_sha256="d" * 64,
            relay_single_token_lexicon_artifact_sha256="e" * 64,
            trd_epistemic_token_ids=[10],
            relay_single_token_ids=[20],
            frozen_scale=scale,
        )
        self.assertFalse(wrong_primary_quantile.production_ready)

        wrong_normalization = calibrate_thresholds(
            [state("norm-a"), state("norm-b", d_l=0.8, s2=0.8)],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            trd_epistemic_lexicon_artifact_sha256="d" * 64,
            relay_single_token_lexicon_artifact_sha256="e" * 64,
            trd_epistemic_token_ids=[10],
            relay_single_token_ids=[20],
            frozen_scale=FrozenScaleArtifact(
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                vocabulary_sha256="f" * 64,
                code_revision="1" * 40,
            ),
        )
        self.assertFalse(wrong_normalization.production_ready)

        constant_scale = FrozenScaleArtifact(
            0.05,
            0.95,
            0.0,
            0.0,
            0.5,
            0.5,
            vocabulary_sha256="f" * 64,
            code_revision="1" * 40,
        )
        constant = calibrate_thresholds(
            [state("constant"), state("constant-2", d_l=0.8, s2=0.8)],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            trd_epistemic_lexicon_artifact_sha256="d" * 64,
            relay_single_token_lexicon_artifact_sha256="e" * 64,
            trd_epistemic_token_ids=[10],
            relay_single_token_ids=[20],
            frozen_scale=constant_scale,
        )
        self.assertFalse(constant.production_ready)
        with self.assertRaises(ValueError):
            constant.assert_production_ready()

    def test_threshold_artifact_hash_detects_tampering(self) -> None:
        scale = FrozenScaleArtifact(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        artifact = calibrate_thresholds(
            [state("a"), state("b", d_l=0.8)],
            normalization_low=0.0,
            normalization_high=1.0,
            lexicon_artifact_sha256="e" * 64,
            reflection_token_ids=[20, 10, 20],
            frozen_scale=scale,
        )
        payload = artifact.to_dict()
        restored = ThresholdArtifact.from_dict(payload)
        self.assertEqual(restored.artifact_sha256, artifact.artifact_sha256)
        self.assertEqual(restored.normalization_low, 0.0)
        self.assertEqual(restored.normalization_high, 1.0)
        self.assertEqual(restored.frozen_scale(), scale)
        self.assertEqual(restored.reflection_token_ids, (10, 20))
        anchor_tamper = artifact.to_dict()
        anchor_tamper.pop("artifact_sha256")
        anchor_tamper["divergence_q_high"] = 2.0
        with self.assertRaises(ValueError):
            ThresholdArtifact.from_dict(anchor_tamper)
        payload["s1_threshold"] = 999.0
        with self.assertRaises(ValueError):
            ThresholdArtifact.from_dict(payload)

    def test_production_threshold_rejects_q80_below_q75_band(self) -> None:
        scale = FrozenScaleArtifact(
            0.05,
            0.95,
            0.0,
            1.0,
            0.0,
            1.0,
            vocabulary_sha256="f" * 64,
            code_revision="1" * 40,
        )
        artifact = calibrate_thresholds(
            [state("a"), state("b", d_l=0.8, s2=0.8)],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            trd_epistemic_lexicon_artifact_sha256="d" * 64,
            relay_single_token_lexicon_artifact_sha256="e" * 64,
            trd_epistemic_token_ids=[10],
            relay_single_token_ids=[20],
            frozen_scale=scale,
        )
        payload = artifact.to_dict()
        payload.pop("artifact_sha256")
        payload["s1_threshold"] = 0.0
        tampered = ThresholdArtifact.from_dict(payload)
        self.assertFalse(tampered.production_ready)

    def test_label_only_a2_permutation_is_prohibited(self) -> None:
        records = [
            state("a", "p1", d_l=0.9),
            state("b", "p1", s2=0.02),
            state("c", "p1"),
            state("d", "p2", d_l=0.9),
            state("e", "p2", s2=0.02),
            state("f", "p2"),
        ]
        original = route_batch(records, self.thresholds)
        with self.assertRaises(RuntimeError):
            permute_actions_within_blocks(original, seed=5, block_by="problem_id")

    def test_a2_bundle_permutation_preserves_lengths_costs_and_payloads(self) -> None:
        assignments = [
            ActionAssignment(
                "a",
                "block",
                Action.REPAIR,
                Action.REPAIR,
                0,
                "a" * 64,
                "b" * 64,
                "not_applicable",
            ),
            ActionAssignment(
                "b",
                "block",
                Action.INTERVENE,
                Action.INTERVENE,
                17,
                "c" * 64,
                "d" * 64,
                "accepted",
                "a" * 64,
            ),
            ActionAssignment(
                "c",
                "block",
                Action.INTERVENE,
                Action.REPAIR,
                11,
                "e" * 64,
                "f" * 64,
                "rejected",
                "b" * 64,
            ),
        ]
        shuffled = permute_action_bundles_within_blocks(assignments, seed=9)
        original_bundles = sorted(
            (
                item.requested_action,
                item.effective_action,
                item.bridge_token_length,
                item.cost_signature,
                item.payload_hash,
                item.gate_status,
                item.gate_artifact_sha256,
            )
            for item in assignments
        )
        shuffled_bundles = sorted(
            (
                item.requested_action,
                item.effective_action,
                item.bridge_token_length,
                item.cost_signature,
                item.payload_hash,
                item.gate_status,
                item.gate_artifact_sha256,
            )
            for item in shuffled
        )
        self.assertEqual(original_bundles, shuffled_bundles)
        self.assertTrue(all(item.state_id != item.source_state_id for item in shuffled))

    def test_a2_bundle_permutation_rejects_singleton_and_duplicate_blocks(self) -> None:
        singleton = ActionAssignment(
            "a",
            "block",
            Action.REPAIR,
            Action.REPAIR,
            0,
            "a" * 64,
            "b" * 64,
            "not_applicable",
        )
        with self.assertRaises(ValueError):
            permute_action_bundles_within_blocks([singleton], seed=1)
        duplicate = ActionAssignment(
            "a",
            "block",
            Action.DISCARD,
            Action.DISCARD,
            0,
            "c" * 64,
            "d" * 64,
            "not_applicable",
        )
        with self.assertRaises(ValueError):
            permute_action_bundles_within_blocks([singleton, duplicate], seed=1)

    def test_d4_repetition_bypass_requires_alternative(self) -> None:
        policy = RouterPolicy(repetition_threshold=0.8)
        with_alternative = state(
            "loop-a",
            repetition_rate=0.9,
            alternative_direction_available=True,
            intervention_budget_available=True,
        )
        without_alternative = state("loop-b", repetition_rate=0.9)
        self.assertEqual(
            route_state(with_alternative, self.thresholds, policy).action,
            Action.INTERVENE,
        )
        self.assertEqual(
            route_state(without_alternative, self.thresholds, policy).action,
            Action.DISCARD,
        )
        without_budget = state(
            "loop-c", repetition_rate=0.9, alternative_direction_available=True
        )
        self.assertEqual(
            route_state(without_budget, self.thresholds, policy).action,
            Action.DISCARD,
        )
        self.assertIn("nonrelay_phi", route_state(with_alternative, self.thresholds, policy).reason)

    def test_d5_zero_pass_rate_rescue_is_opt_in(self) -> None:
        record = state("zero", p_hat=0.0)
        self.assertEqual(route_state(record, self.thresholds).action, Action.DISCARD)
        self.assertEqual(
            route_state(record, self.thresholds, RouterPolicy(paced_zero_rescue=True)).action,
            Action.DISCARD,
        )
        self.assertEqual(
            route_state(
                state(
                    "zero-eligible",
                    p_hat=0.0,
                    relay_phi_eligible=True,
                    intervention_budget_available=True,
                    intervention_cooldown_available=True,
                ),
                self.thresholds,
                RouterPolicy(paced_zero_rescue=True),
            ).action,
            Action.INTERVENE,
        )
        self.assertEqual(
            route_state(
                state("zero-repair", p_hat=0.0, d_l=0.8),
                self.thresholds,
                RouterPolicy(paced_zero_rescue=True),
            ).action,
            Action.REPAIR,
        )


if __name__ == "__main__":
    unittest.main()
