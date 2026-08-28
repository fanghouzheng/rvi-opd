import math
import unittest

from rvi_opd.models import RawStateSignal
from rvi_opd.signals import (
    EPISTEMIC_ONSET_PHRASES,
    FrozenScaleArtifact,
    RELAY_REFLECTION_PHRASES,
    apply_frozen_scale,
    decompose_batch,
    decompose_grouped_batches,
    fit_frozen_scale,
    handoff_trigger,
    local_support_signal,
    quantile,
    reflection_mass,
    robust_normalize,
)


class SignalTests(unittest.TestCase):
    def test_local_support_uses_union_and_forward_kl(self) -> None:
        teacher = {"a": 0.20, "b": 0.70, "c": 0.10}
        student = {"a": 0.80, "b": 0.10, "c": 0.10}
        divergence, compatibility = local_support_signal(teacher, student, k=1)
        expected = (0.20 / 0.90) * math.log((0.20 / 0.90) / (0.80 / 0.90))
        expected += (0.70 / 0.90) * math.log((0.70 / 0.90) / (0.10 / 0.90))
        self.assertAlmostEqual(divergence, expected)
        self.assertAlmostEqual(compatibility, 0.20)

    def test_local_support_keeps_absolute_coverage_on_gathered_subset(self) -> None:
        teacher = {"a": 0.20, "b": 0.70}
        student = {"a": 0.80, "b": 0.10}
        divergence, compatibility = local_support_signal(teacher, student, k=1)
        expected = (0.20 / 0.90) * math.log((0.20 / 0.90) / (0.80 / 0.90))
        expected += (0.70 / 0.90) * math.log((0.70 / 0.90) / (0.10 / 0.90))
        self.assertAlmostEqual(divergence, expected)
        self.assertAlmostEqual(compatibility, 0.20)

    def test_robust_normalization_clips_tails(self) -> None:
        values = [0.0, 1.0, 2.0, 100.0]
        normalized = robust_normalize(values, q_low=0.25, q_high=0.75)
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[-1], 1.0)
        self.assertTrue(0.0 < normalized[1] < normalized[2] < 1.0)

    def test_decomposition_is_complementary(self) -> None:
        records = [
            RawStateSignal("a", "p", "t", 0, 0.1, 0.2, 0.001),
            RawStateSignal("b", "p", "t", 1, 0.5, 0.8, 0.002),
            RawStateSignal("c", "p", "t", 2, 1.0, 0.4, 0.003),
        ]
        for record in decompose_batch(records, q_low=0.0, q_high=1.0):
            self.assertAlmostEqual(record.d_l + record.d_i, record.d_tilde)

    def test_reflection_mass_deduplicates_token_ids(self) -> None:
        distribution = {1: 0.1, 2: 0.2, 3: 0.7}
        self.assertAlmostEqual(reflection_mass(distribution, [1, 2, 2]), 0.3)

    def test_reflection_mass_is_absolute_on_gathered_subset(self) -> None:
        self.assertAlmostEqual(reflection_mass({1: 0.006, 2: 0.094}, [1]), 0.006)

    def test_reflection_mass_rejects_empty_ids(self) -> None:
        with self.assertRaises(ValueError):
            reflection_mass({1: 1.0}, [])

    def test_reflection_mass_rejects_missing_topk_probability(self) -> None:
        with self.assertRaises(ValueError):
            reflection_mass({1: 0.4, 2: 0.6}, [1, 99])

    def test_local_support_rejects_incomplete_union_scores(self) -> None:
        with self.assertRaises(ValueError):
            local_support_signal({"a": 0.8, "b": 0.2}, {"a": 1.0}, k=1)

    def test_signal_probabilities_must_be_absolute_not_unnormalized_scores(self) -> None:
        with self.assertRaises(ValueError):
            local_support_signal({"a": 0.8, "b": 0.8}, {"a": 0.8, "b": 0.8}, k=1)

    def test_handoff_trigger_matches_teacher_student_asymmetry(self) -> None:
        teacher = {"Wait": 0.7, "So": 0.2, "x": 0.1}
        student = {"Wait": 0.01, "So": 0.8, "x": 0.19}
        self.assertTrue(handoff_trigger(teacher, student, {"Wait"}, k=2))
        self.assertFalse(handoff_trigger(teacher, student, {"Wait"}, k=3))

    def test_handoff_requires_same_vocabulary_and_complete_reflection_scores(self) -> None:
        with self.assertRaises(ValueError):
            handoff_trigger(
                {"Wait": 0.7, "x": 0.3},
                {"Wait": 0.1, "y": 0.9},
                {"Wait"},
            )
        with self.assertRaises(ValueError):
            handoff_trigger({"x": 1.0}, {"x": 1.0}, {"Wait"})
        with self.assertRaises(ValueError):
            handoff_trigger({"x": 1.0}, {"x": 1.0}, set())

    def test_quantile_interpolates(self) -> None:
        self.assertAlmostEqual(quantile([0.0, 10.0], 0.25), 2.5)

    def test_trd_and_relay_lexicons_are_not_conflated(self) -> None:
        self.assertEqual(len(EPISTEMIC_ONSET_PHRASES), 16)
        self.assertEqual(len(RELAY_REFLECTION_PHRASES), 13)
        self.assertNotEqual(set(EPISTEMIC_ONSET_PHRASES), set(RELAY_REFLECTION_PHRASES))

    def test_grouped_batches_normalize_independently_and_keep_order(self) -> None:
        records = [
            RawStateSignal("a1", "p", "t", 0, 0.0, 0.0, 0.0, batch_id="a"),
            RawStateSignal("b1", "p", "t", 0, 100.0, 0.0, 0.0, batch_id="b"),
            RawStateSignal("a2", "p", "t", 1, 1.0, 1.0, 0.0, batch_id="a"),
            RawStateSignal("b2", "p", "t", 1, 200.0, 1.0, 0.0, batch_id="b"),
        ]
        output = decompose_grouped_batches(records, q_low=0.0, q_high=1.0)
        self.assertEqual([item.state_id for item in output], ["a1", "b1", "a2", "b2"])
        for actual, expected in zip(
            [item.d_tilde for item in output], [0.0, 0.0, 1.0, 1.0]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_frozen_scale_is_independent_of_batch_id_and_batch_peers(self) -> None:
        calibration = [
            RawStateSignal("cal-low", "p", "t", 0, 0.1, 0.1, 0.0),
            RawStateSignal("cal-high", "p", "t", 1, 1.1, 0.9, 0.0),
        ]
        scale = fit_frozen_scale(calibration, q_low=0.0, q_high=1.0)
        first = apply_frozen_scale(
            [
                RawStateSignal(
                    "target", "p", "t", 2, 0.6, 0.75, 0.004, batch_id="batch-a"
                ),
                RawStateSignal("peer-a", "p", "t", 3, 0.0, 0.0, 0.0),
            ],
            scale,
        )[0]
        second = apply_frozen_scale(
            [
                RawStateSignal(
                    "target", "p", "t", 2, 0.6, 0.75, 0.004, batch_id="batch-z"
                ),
                RawStateSignal("peer-z", "p", "t", 3, 100.0, 1.0, 0.0),
            ],
            scale,
        )[0]
        self.assertEqual(first.batch_id, "batch-a")
        self.assertEqual(second.batch_id, "batch-z")
        self.assertEqual(
            (first.d_tilde, first.c_tilde, first.d_l, first.d_i, first.s1, first.s2),
            (second.d_tilde, second.c_tilde, second.d_l, second.d_i, second.s1, second.s2),
        )

    def test_frozen_scale_hash_detects_tampering(self) -> None:
        scale = FrozenScaleArtifact(0.0, 1.0, 0.1, 1.1, 0.2, 0.8)
        payload = scale.to_dict()
        self.assertEqual(
            FrozenScaleArtifact.from_dict(payload).artifact_sha256,
            scale.artifact_sha256,
        )
        payload["divergence_q_high"] = 2.0
        with self.assertRaises(ValueError):
            FrozenScaleArtifact.from_dict(payload)

    def test_raw_state_rejects_nonfinite_or_out_of_range_signals(self) -> None:
        for kwargs in (
            {"divergence": float("nan")},
            {"compatibility": 1.1},
            {"s2": -0.1},
            {"token_index": -1},
        ):
            row = {
                "state_id": "a",
                "problem_id": "p",
                "trajectory_id": "t",
                "token_index": 0,
                "divergence": 0.1,
                "compatibility": 0.2,
                "s2": 0.01,
                **kwargs,
            }
            with self.assertRaises(ValueError):
                RawStateSignal(**row)


if __name__ == "__main__":
    unittest.main()
