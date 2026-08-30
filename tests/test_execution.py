import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rvi_opd.execution import (
    config_bundle_sha256,
    evaluate_healthbench_gate,
    execution_readiness,
    load_json_object,
    sha256_file,
    validate_execution_policy,
)


class HealthBenchFirstExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.policy_path = cls.root / "configs/execution/healthbench-first.json"
        cls.policy = load_json_object(cls.policy_path)
        cls.code_revision = subprocess.run(
            ["git", "-C", str(cls.root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def valid_result(self):
        core = self.policy["healthbench_core"]
        bindings = self.policy["config_bindings"]
        training_runs = [
            {
                "arm": arm,
                "seed": seed,
                "run_manifest_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
            }
            for arm in core["trained_arms"]
            for seed in core["training_seeds"]
        ]
        evaluation_manifests = [
            {
                "arm": arm,
                "sampling_seed": seed,
                "completion_manifest_sha256": "c" * 64,
                "grader_manifest_sha256": "d" * 64,
            }
            for arm, seeds in core["evaluation_sampling_seeds"].items()
            for seed in seeds
        ]

        def simultaneous_comparisons(arms):
            return {
                arm: {
                    "estimate": 0.02,
                    "simultaneous_lower_95ci": 0.005,
                    "upper_95ci": 0.035,
                }
                for arm in arms
            }

        return {
            "schema": self.policy["gate_result_schema"],
            "policy_sha256": sha256_file(self.policy_path),
            "code_revision": self.code_revision,
            "e2_config_sha256": sha256_file(self.root / bindings["e2_config"]),
            "math_config_bundle_sha256": config_bundle_sha256(
                self.root, bindings["math_configs"]
            ),
            "flags": {
                name: True for name in self.policy["required_true_flags"]
            },
            "training_runs": training_runs,
            "evaluation_manifests": evaluation_manifests,
            "counts": {"full_prompt_count": 5000, "hard_prompt_count": 1000},
            "metrics": {
                "rvi_vs_base": {
                    "estimate": 0.02,
                    "lower_95ci": 0.005,
                    "upper_95ci": 0.035,
                },
                "rvi_vs_non_oracle": simultaneous_comparisons(
                    self.policy["decision_rule"]["non_oracle_baselines"]
                ),
                "rvi_vs_single_actions": simultaneous_comparisons(
                    self.policy["decision_rule"]["single_action_baselines"]
                ),
                "rvi_vs_a2": {
                    "estimate": 0.02,
                    "lower_95ci": 0.005,
                    "upper_95ci": 0.035,
                    "leave_one_seed_out_estimates": [0.012, 0.018, 0.015],
                },
                "rubric_mechanism": {
                    "did_holm_adjusted_lower_95ci": 0.002,
                    "repair_insertable_lower_95ci": 0.002,
                    "intervene_insertable_lower_95ci": 0.003,
                    "intervene_global_revision_lower_95ci": 0.002,
                    "repair_global_revision_90ci": {
                        "lower": -0.005,
                        "upper": 0.006,
                    },
                },
                "negative_violation_noninferiority": {
                    comparator: {"upper_95ci": 0.005}
                    for comparator in [
                        "base",
                        *self.policy["decision_rule"]["non_oracle_baselines"],
                    ]
                },
            },
        }

    def write_result(self, directory: str, payload) -> Path:
        path = Path(directory) / "healthbench-gate.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_policy_is_strict_and_covers_each_target_once(self) -> None:
        self.assertEqual(validate_execution_policy(self.policy, self.root), [])
        changed = copy.deepcopy(self.policy)
        changed["math_targets"].remove("D1:math")
        changed["unknown"] = True
        errors = "\n".join(validate_execution_policy(changed, self.root))
        self.assertIn("contains unknown fields: unknown", errors)
        self.assertIn("math target set is not the frozen set", errors)
        self.assertIn("cover every target exactly once", errors)

    def test_medical_is_ready_while_math_waits_for_gate(self) -> None:
        medical = execution_readiness(
            self.policy_path, "D1:medical", enforce_clean_checkout=False
        )
        math = execution_readiness(
            self.policy_path, "D1:math", enforce_clean_checkout=False
        )
        self.assertTrue(medical["order_allowed"])
        self.assertEqual(medical["status"], "ORDER_ALLOWED_PRE_GATE")
        self.assertFalse(medical["run_readiness_assessed"])
        self.assertEqual(medical["scope"], "execution_order_only")
        self.assertFalse(math["order_allowed"])
        self.assertEqual(math["status"], "ORDER_BLOCKED_PENDING_HEALTHBENCH_GATE")

    def test_clean_checkout_is_enforced_for_pre_gate_targets(self) -> None:
        with mock.patch(
            "rvi_opd.execution._assert_clean_checkout",
            side_effect=ValueError("dirty checkout"),
        ) as clean_check:
            with self.assertRaisesRegex(ValueError, "dirty checkout"):
                execution_readiness(
                    self.policy_path,
                    "D1:medical",
                    enforce_clean_checkout=True,
                )
        clean_check.assert_called_once_with(self.root)

    def test_all_frozen_healthbench_signals_release_math(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, self.valid_result())
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
            readiness = execution_readiness(
                self.policy_path,
                "E1",
                result_path,
                enforce_clean_checkout=False,
            )
        self.assertTrue(gate["launch_math"])
        self.assertEqual(gate["decision"], "GO_MATH")
        self.assertTrue(all(check["passed"] for check in gate["checks"]))
        self.assertTrue(readiness["order_allowed"])
        self.assertEqual(
            readiness["status"], "ORDER_ALLOWED_MATH_AFTER_HEALTHBENCH"
        )

    def test_a2_instability_stops_math_even_when_other_metrics_pass(self) -> None:
        payload = self.valid_result()
        payload["metrics"]["rvi_vs_a2"]["leave_one_seed_out_estimates"][1] = -0.001
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        self.assertFalse(gate["launch_math"])
        self.assertEqual(gate["decision"], "STOP_AFTER_HEALTHBENCH")
        failed = {check["name"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("a2_leave_one_seed_out_stability", failed)

    def test_beating_trained_arms_does_not_release_math_when_rvi_loses_to_base(self) -> None:
        payload = self.valid_result()
        payload["metrics"]["rvi_vs_base"] = {
            "estimate": -0.005,
            "lower_95ci": -0.012,
            "upper_95ci": 0.002,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        self.assertFalse(gate["launch_math"])
        failed = {check["name"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("rvi_beats_base", failed)
        self.assertIn("rvi_material_gain_over_base", failed)

    def test_missing_run_and_false_readiness_flag_fail_closed(self) -> None:
        payload = self.valid_result()
        payload["training_runs"].pop()
        payload["flags"]["all_required_runs_complete"] = False
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        self.assertFalse(gate["launch_math"])
        failed = {check["name"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("complete_training_run_set", failed)
        self.assertIn("flag:all_required_runs_complete", failed)

    def test_malformed_artifact_hash_stops_math(self) -> None:
        payload = self.valid_result()
        payload["training_runs"][0]["checkpoint_sha256"] = "not-a-sha256"
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        self.assertFalse(gate["launch_math"])
        failed = {check["name"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("training_artifact_hashes", failed)

    def test_stale_config_binding_is_rejected(self) -> None:
        payload = self.valid_result()
        payload["math_config_bundle_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            with self.assertRaisesRegex(ValueError, "math config bundle"):
                evaluate_healthbench_gate(
                    self.policy_path, result_path, enforce_clean_checkout=False
                )

    def test_stale_policy_code_and_e2_bindings_are_rejected(self) -> None:
        cases = (
            ("policy_sha256", "0" * 64, "policy SHA256"),
            ("code_revision", "0" * 40, "code_revision"),
            ("e2_config_sha256", "0" * 64, "E2 config SHA256"),
        )
        for field, value, message in cases:
            payload = self.valid_result()
            payload[field] = value
            with tempfile.TemporaryDirectory() as temporary:
                result_path = self.write_result(temporary, payload)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_healthbench_gate(
                        self.policy_path,
                        result_path,
                        enforce_clean_checkout=False,
                    )

    def test_metric_threshold_boundaries_follow_the_frozen_rules(self) -> None:
        payload = self.valid_result()
        for family in ("rvi_vs_non_oracle", "rvi_vs_single_actions"):
            for interval in payload["metrics"][family].values():
                interval["estimate"] = 0.01
                interval["simultaneous_lower_95ci"] = 0.001
                interval["upper_95ci"] = 0.02
        for value in payload["metrics"]["negative_violation_noninferiority"].values():
            value["upper_95ci"] = 0.01
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        self.assertTrue(gate["launch_math"], "material >= and safety <= are inclusive")

        payload = self.valid_result()
        payload["metrics"]["rvi_vs_non_oracle"]["vanilla_opd"][
            "simultaneous_lower_95ci"
        ] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            gate = evaluate_healthbench_gate(
                self.policy_path, result_path, enforce_clean_checkout=False
            )
        failed = {check["name"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("rvi_beats_each_non_oracle", failed)

    def test_tost_interval_must_be_strictly_inside_equivalence_margin(self) -> None:
        for boundary in (-0.01, 0.01):
            payload = self.valid_result()
            repair_ci = payload["metrics"]["rubric_mechanism"][
                "repair_global_revision_90ci"
            ]
            if boundary < 0:
                repair_ci["lower"] = boundary
            else:
                repair_ci["upper"] = boundary
            with tempfile.TemporaryDirectory() as temporary:
                result_path = self.write_result(temporary, payload)
                gate = evaluate_healthbench_gate(
                    self.policy_path, result_path, enforce_clean_checkout=False
                )
            failed = {
                check["name"] for check in gate["checks"] if not check["passed"]
            }
            self.assertIn("repair_global_revision_tost", failed)

    def test_safety_nested_schema_and_counts_fail_closed(self) -> None:
        payload = self.valid_result()
        payload["metrics"]["negative_violation_noninferiority"]["base"][
            "estimate"
        ] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                evaluate_healthbench_gate(
                    self.policy_path, result_path, enforce_clean_checkout=False
                )

        payload = self.valid_result()
        payload["counts"]["full_prompt_count"] = 5000.0
        with tempfile.TemporaryDirectory() as temporary:
            result_path = self.write_result(temporary, payload)
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                evaluate_healthbench_gate(
                    self.policy_path, result_path, enforce_clean_checkout=False
                )


if __name__ == "__main__":
    unittest.main()
