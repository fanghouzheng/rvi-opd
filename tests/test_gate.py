import unittest

from rvi_opd.gate import (
    FrozenJointGateArtifact,
    GateConfig,
    evaluate_intervention_gate,
    evaluate_joint_intervention_gate,
    fit_joint_gate_artifact,
)
from rvi_opd.models import Action


class GateTests(unittest.TestCase):
    def test_accepts_s2_drop(self) -> None:
        decision = evaluate_intervention_gate(
            [0.010] * 4,
            [0.004] * 4,
            [0.2] * 4,
            [0.2] * 4,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.effective_action, Action.INTERVENE)

    def test_accepts_teacher_preferred_gain(self) -> None:
        decision = evaluate_intervention_gate(
            [0.010] * 4,
            [0.010] * 4,
            [0.2] * 4,
            [0.3] * 4,
        )
        self.assertTrue(decision.accepted)

    def test_rolls_back_to_repair(self) -> None:
        decision = evaluate_intervention_gate(
            [0.010] * 4,
            [0.010] * 4,
            [0.2] * 4,
            [0.21] * 4,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.requested_action, Action.INTERVENE)
        self.assertEqual(decision.effective_action, Action.REPAIR)

    def test_rejects_unpaired_probes(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_intervention_gate(
                [0.1] * 4,
                [0.1] * 3,
                [0.1] * 4,
                [0.1] * 4,
                GateConfig(min_rollouts=3),
            )

    def test_rejects_invalid_config_and_metrics(self) -> None:
        with self.assertRaises(ValueError):
            GateConfig(min_rollouts=0)
        with self.assertRaises(ValueError):
            GateConfig(min_s2_drop=float("nan"))
        with self.assertRaises(ValueError):
            evaluate_intervention_gate(
                [float("nan")] * 4,
                [0.1] * 4,
                [0.1] * 4,
                [0.1] * 4,
            )

    def test_joint_max_gate_uses_one_frozen_event_threshold(self) -> None:
        artifact = FrozenJointGateArtifact(
            s2_drop_null_mean=0.0,
            s2_drop_null_std=0.1,
            agreement_gain_null_mean=0.0,
            agreement_gain_null_std=0.1,
            max_stat_q95=2.0,
            record_count=100,
        )
        accepted = evaluate_joint_intervention_gate(
            [0.8] * 4,
            [0.5] * 4,
            [0.2] * 4,
            [0.2] * 4,
            artifact,
        )
        rejected = evaluate_joint_intervention_gate(
            [0.5] * 4,
            [0.4] * 4,
            [0.2] * 4,
            [0.3] * 4,
            artifact,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.gate_mode, "joint_max_statistic")
        self.assertAlmostEqual(accepted.gate_statistic, 3.0)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.effective_action, Action.REPAIR)

    def test_joint_gate_artifact_fit_hash_and_production_metadata(self) -> None:
        artifact = fit_joint_gate_artifact(
            [-0.1, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.0, -0.1],
            calibration_split_sha256="a" * 64,
            threshold_artifact_sha256="b" * 64,
            code_revision="c" * 40,
        )
        artifact.assert_production_ready()
        payload = artifact.to_dict()
        restored = FrozenJointGateArtifact.from_dict(payload)
        self.assertEqual(restored.artifact_sha256, artifact.artifact_sha256)
        payload["max_stat_q95"] = 999.0
        with self.assertRaises(ValueError):
            FrozenJointGateArtifact.from_dict(payload)

    def test_joint_gate_fit_rejects_degenerate_null(self) -> None:
        with self.assertRaises(ValueError):
            fit_joint_gate_artifact([0.0, 0.0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
