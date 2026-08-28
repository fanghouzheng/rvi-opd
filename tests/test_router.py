import unittest
from collections import Counter

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
    )


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ThresholdArtifact(0.5, 0.01, 0.8, 0.8, "fixture", 10)

    def test_s2_takes_precedence(self) -> None:
        decision = route_state(state("x", d_l=0.9, s2=0.02), self.thresholds)
        self.assertEqual(decision.action, Action.INTERVENE)

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
            [state("a")],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            lexicon_artifact_sha256="e" * 64,
            reflection_token_ids=[10, 20],
            frozen_scale=scale,
        )
        ready.assert_production_ready()
        self.assertEqual(ready.reflection_token_ids, (10, 20))
        self.assertEqual(ready.scale_artifact_sha256, scale.artifact_sha256)
        self.assertEqual(ready.vocabulary_sha256, "f" * 64)
        self.assertEqual(ready.code_revision, "1" * 40)

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
            [state("constant")],
            teacher_revision="a" * 40,
            student_revision="b" * 40,
            tokenizer_sha256="c" * 64,
            calibration_split_sha256="d" * 64,
            lexicon_artifact_sha256="e" * 64,
            reflection_token_ids=[10],
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

    def test_a2_preserves_action_counts_within_problem(self) -> None:
        records = [
            state("a", "p1", d_l=0.9),
            state("b", "p1", s2=0.02),
            state("c", "p1"),
            state("d", "p2", d_l=0.9),
            state("e", "p2", s2=0.02),
            state("f", "p2"),
        ]
        original = route_batch(records, self.thresholds)
        shuffled = permute_actions_within_blocks(original, seed=5, block_by="problem_id")
        self.assertTrue(any(a.action != b.action for a, b in zip(original, shuffled)))
        for problem_id in {"p1", "p2"}:
            before = Counter(item.action for item in original if item.problem_id == problem_id)
            after = Counter(item.action for item in shuffled if item.problem_id == problem_id)
            self.assertEqual(before, after)

    def test_a2_bundle_permutation_preserves_lengths_costs_and_payloads(self) -> None:
        assignments = [
            ActionAssignment("a", "block", Action.REPAIR, 0, "cost-r", "hash-r"),
            ActionAssignment("b", "block", Action.INTERVENE, 17, "cost-i", "hash-i"),
            ActionAssignment("c", "block", Action.DISCARD, 0, "cost-d", "hash-d"),
        ]
        shuffled = permute_action_bundles_within_blocks(assignments, seed=9)
        original_bundles = sorted(
            (item.action, item.bridge_token_length, item.cost_signature, item.payload_hash)
            for item in assignments
        )
        shuffled_bundles = sorted(
            (item.action, item.bridge_token_length, item.cost_signature, item.payload_hash)
            for item in shuffled
        )
        self.assertEqual(original_bundles, shuffled_bundles)
        self.assertTrue(all(item.state_id != item.source_state_id for item in shuffled))

    def test_d4_repetition_bypass_requires_alternative(self) -> None:
        policy = RouterPolicy(repetition_threshold=0.8)
        with_alternative = state(
            "loop-a", repetition_rate=0.9, alternative_direction_available=True
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

    def test_d5_zero_pass_rate_rescue_is_opt_in(self) -> None:
        record = state("zero", p_hat=0.0)
        self.assertEqual(route_state(record, self.thresholds).action, Action.DISCARD)
        self.assertEqual(
            route_state(record, self.thresholds, RouterPolicy(paced_zero_rescue=True)).action,
            Action.INTERVENE,
        )


if __name__ == "__main__":
    unittest.main()
