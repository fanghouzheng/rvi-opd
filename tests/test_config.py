import json
import math
import tempfile
import unittest
from pathlib import Path

from rvi_opd.config import (
    validate_config_paths,
    validate_experiment_config,
    validate_upstreams_lock,
)


def valid_config():
    return {
        "id": "D-test",
        "question": "Does the contract hold?",
        "split": "synthetic_only",
        "matrix_revision": "2026-08-01",
        "citation_cutoff": "2026-08-01",
        "concurrent_policy": "post_cutoff_treat_as_concurrent_do_not_cite_or_compare",
        "arms": ["repair", "intervene"],
        "primary_endpoints": ["score"],
        "audit_invariants": ["same_seed"],
    }


def repository_config(filename):
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / filename).read_text(encoding="utf-8"))


class ConfigValidationTests(unittest.TestCase):
    def test_repository_configs_are_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        errors = validate_config_paths(sorted((root / "configs").glob("*.json")))
        self.assertEqual(errors, [])

    def test_required_fields_have_strict_types_and_non_empty_values(self) -> None:
        payload = valid_config()
        payload.update(
            {
                "id": " ",
                "question": 7,
                "split": "",
                "arms": "repair",
                "primary_endpoints": [],
            }
        )
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("fixture: id must be a non-empty string", errors)
        self.assertIn("fixture: question must be a non-empty string", errors)
        self.assertIn("fixture: split must be a non-empty string", errors)
        self.assertIn("fixture: arms must be a non-empty list", errors)
        self.assertIn("fixture: primary_endpoints must be a non-empty list", errors)

    def test_string_lists_reject_wrong_items_and_duplicates(self) -> None:
        payload = valid_config()
        payload["arms"] = ["repair", "repair", ""]
        payload["audit_invariants"] = ["same_seed", 3]
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("fixture: arms contains duplicate value 'repair'", errors)
        self.assertIn("fixture: arms[2] must be a non-empty string", errors)
        self.assertIn("fixture: audit_invariants[1] must be a non-empty string", errors)

    def test_nested_numbers_are_finite_non_negative_and_probabilities_are_bounded(self) -> None:
        payload = valid_config()
        payload["training"] = {
            "optimizer_steps": -1,
            "learning_rate": math.inf,
            "success_rate": 1.01,
            "robust_normalization_quantiles": [0.05, 1.2],
        }
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("fixture: training.optimizer_steps must be non-negative", errors)
        self.assertIn("fixture: training.learning_rate must be finite", errors)
        self.assertIn("fixture: training.success_rate must be in [0, 1]", errors)
        self.assertIn(
            "fixture: training.robust_normalization_quantiles[1] must be in [0, 1]",
            errors,
        )

    def test_probability_fields_require_numeric_values(self) -> None:
        payload = valid_config()
        payload["training"] = {"success_rate": "high", "top_p": True}
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("fixture: training.success_rate must be numeric and in [0, 1]", errors)
        self.assertIn("fixture: training.top_p must be numeric and in [0, 1]", errors)

    def test_top_level_must_be_an_object(self) -> None:
        self.assertEqual(
            validate_experiment_config([], "fixture"),
            ["fixture: top-level config must be an object"],
        )

    def test_nested_empty_values_and_null_are_rejected(self) -> None:
        payload = valid_config()
        payload["metadata"] = {"owner": "", "tags": [], "revision": None}
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("fixture: metadata.owner must be a non-empty string", errors)
        self.assertIn("fixture: metadata.tags must be a non-empty list", errors)
        self.assertIn("fixture: metadata.revision must not be null", errors)

    def test_duplicate_ids_across_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = directory / "first.json"
            second = directory / "second.json"
            first.write_text(json.dumps(valid_config()), encoding="utf-8")
            second.write_text(json.dumps(valid_config()), encoding="utf-8")
            errors = "\n".join(validate_config_paths([first, second]))
        self.assertIn("duplicate experiment id 'D-test'", errors)
        self.assertIn(str(first), errors)

    def test_final_matrix_semantics_are_checked(self) -> None:
        payload = valid_config()
        payload["id"] = "D0"
        payload["design"] = {
            "factors": {"signal_type": ["dl_top", "di_top"], "s2_band": ["low", "high"], "action": ["repair", "intervene"]},
            "minimum_states": 4,
            "minimum_states_per_cell": 1,
        }
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("D0 factors must be exactly", errors)

    def test_lock_revision_cross_check(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = root / "upstreams.lock.json"
        payload = {
            "models": {
                "teacher": "Qwen/Qwen3-4B-Instruct-2507",
                "teacher_revision": "0" * 40,
            }
        }
        errors = "\n".join(validate_upstreams_lock(lock, [payload]))
        self.assertIn("disagrees with lock", errors)

    def test_lock_checks_train_alias_and_official_evaluation_dataset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = root / "upstreams.lock.json"
        payload = {
            "data": {
                "train": "BytedTsinghua-SIA/DAPO-Math-17k",
                "revision": "0" * 40,
            },
            "official_evaluation": {
                "dataset": "openai/healthbench",
                "revision": "1" * 40,
            },
        }
        errors = "\n".join(validate_upstreams_lock(lock, [payload]))
        self.assertIn("data: revision", errors)
        self.assertIn("official_evaluation: revision", errors)

    def test_d1_primary_router_quantiles_are_frozen_at_q80(self) -> None:
        payload = valid_config()
        payload["id"] = "D1"
        payload["signals"] = {
            "router_threshold_scope": "global_frozen",
            "router_primary_quantiles": {"s1": 0.75, "s2": 0.75},
        }
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("must freeze s1/s2 at q80", errors)

    def test_known_experiment_ids_fail_closed_when_their_sections_are_missing(self) -> None:
        required_sections = {
            "ablations.json": "arm_contracts",
            "d0_factorial.json": "design",
            "d1_signal_calibration.json": "signals",
            "d2_paired_continuation.json": "continuation_protocol",
            "d3_detached.json": "detached_contract",
            "d4_degenerate_prefix.json": "challenge_selection",
            "d5_paced_rescue.json": "selection_artifact",
            "e1_math.json": "evaluation",
            "e2_healthbench.json": "official_evaluation",
        }
        for filename, section in required_sections.items():
            with self.subTest(filename=filename, section=section):
                payload = repository_config(filename)
                del payload[section]
                errors = "\n".join(validate_experiment_config(payload, "fixture"))
                self.assertIn("missing experiment fields", errors)
                self.assertIn(section, errors)

    def test_d0_contract_requires_exact_2x2_arms_and_s2_strata(self) -> None:
        payload = repository_config("d0_factorial.json")
        payload["arms"][-1] = "di_top_defer"
        payload["design"]["factors"]["s2_band"] = ["low", "high"]
        payload["design"]["s2_analysis_strata"]["low"] = "bottom_half"
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("D0.arms must contain the frozen arm set", errors)
        self.assertIn("D0.design.factors contains unknown fields: s2_band", errors)
        self.assertIn("D0.design.s2_analysis_strata.low must equal", errors)

    def test_d1_contract_freezes_probe_aggregation_and_one_way_artifact_binding(self) -> None:
        payload = repository_config("d1_signal_calibration.json")
        payload["acceptance_gate_calibration"]["probe_rollouts_per_event"] = 3
        payload["acceptance_gate_calibration"]["probe_aggregation"] = "median"
        payload["artifact_schema"]["version"] = "rvi-d1-freeze-bundle-v2"
        payload["artifact_schema"]["threshold_artifact"]["required_hash_fields"].append(
            "gate_artifact_sha256"
        )
        payload["artifact_schema"]["joint_gate_artifact"]["one_way_binding"] = (
            "threshold_and_gate_bind_each_other"
        )
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("probe_rollouts_per_event must equal 4", errors)
        self.assertIn("probe_aggregation must equal 'paired_arithmetic_mean'", errors)
        self.assertIn("artifact_schema.version must equal 'rvi-d1-freeze-bundle-v3'", errors)
        self.assertIn("threshold artifact must not bind the gate hash", errors)
        self.assertIn("joint_gate_artifact.one_way_binding must equal", errors)

    def test_ablation_contract_freezes_real_budget_axis_and_cut_order(self) -> None:
        payload = repository_config("ablations.json")
        payload["parameter_grids"]["A5"][
            "teacher_gpu_seconds_per_prompt_relative_to_frozen_default"
        ] = [0.5, 1.0]
        payload["cut_first_order"] = ["A6", "D5"]
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn(
            "teacher_gpu_seconds_per_prompt_relative_to_frozen_default must equal",
            errors,
        )
        self.assertIn("cut_first_order must equal", errors)

    def test_e1_contract_freezes_benchmarks_arms_seeds_and_generation(self) -> None:
        payload = repository_config("e1_math.json")
        payload["evaluation"]["benchmarks_32_samples"][0] = "AIME2023"
        payload["main_table_arms"].remove("rvi_opd")
        payload["seed_policy"]["core_five_seeds"] = [13, 17, 23]
        payload["training"]["rollout_temperature"] = 0.7
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("E1.main_table_arms must contain the frozen arm set", errors)
        self.assertIn("E1.evaluation.benchmarks_32_samples must equal", errors)
        self.assertIn("E1.seed_policy.core_five_seeds must equal", errors)
        self.assertIn("E1.training.rollout_temperature must equal 1.0", errors)

    def test_e2_contract_requires_rvi_main_arm_and_fixed_sampling(self) -> None:
        payload = repository_config("e2_healthbench.json")
        payload["main_table_arms"].remove("rvi_opd")
        payload["training"]["max_response_tokens"] = True
        payload["official_evaluation"]["answer_generation"]["top_k"] = 50
        payload["official_evaluation"]["staged_evaluation"]["pilot_prerequisite"] = (
            "pilot_before_annotation_freeze"
        )
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("E2.main_table_arms must contain the frozen arm set", errors)
        self.assertIn("E2.training.max_response_tokens must be an integer", errors)
        self.assertIn("E2.official_evaluation.answer_generation.top_k must equal 0", errors)
        self.assertIn("pilot_prerequisite must equal", errors)

    def test_boolean_cannot_masquerade_as_a_generic_integer(self) -> None:
        payload = valid_config()
        payload["training"] = {
            "optimizer_steps": True,
            "stop_token_ids": [151645, False],
            "enabled": True,
        }
        errors = "\n".join(validate_experiment_config(payload, "fixture"))
        self.assertIn("training.optimizer_steps must be an integer, not a boolean", errors)
        self.assertIn("training.stop_token_ids[1] must be an integer, not a boolean", errors)
        self.assertNotIn("training.enabled", errors)

    def test_unknown_fields_are_rejected_in_result_defining_sections(self) -> None:
        e1 = repository_config("e1_math.json")
        e1["training"]["temperatur"] = 1.0
        e2 = repository_config("e2_healthbench.json")
        e2["official_evaluation"]["answer_generation"]["top_pp"] = 1.0
        d0 = repository_config("d0_factorial.json")
        d0["design"]["third_factor"] = {"levels": ["low", "high"]}
        d0["floating_override"] = True
        self.assertIn(
            "E1.training contains unknown fields: temperatur",
            "\n".join(validate_experiment_config(e1, "fixture")),
        )
        self.assertIn(
            "E2.official_evaluation.answer_generation contains unknown fields: top_pp",
            "\n".join(validate_experiment_config(e2, "fixture")),
        )
        self.assertIn(
            "D0.design contains unknown fields: third_factor",
            "\n".join(validate_experiment_config(d0, "fixture")),
        )
        self.assertIn(
            "D0 contains unknown top-level fields: floating_override",
            "\n".join(validate_experiment_config(d0, "fixture")),
        )

    def test_run_ready_mode_rejects_hash_and_revision_placeholders(self) -> None:
        payload = valid_config()
        payload["artifacts"] = {
            "manifest_sha256": "required_before_first_run",
            "model_revision": "required_before_training",
        }
        self.assertEqual(validate_experiment_config(payload, "fixture"), [])

        errors = "\n".join(
            validate_experiment_config(payload, "fixture", run_ready=True)
        )
        self.assertIn("manifest_sha256 is not run-ready", errors)
        self.assertIn("model_revision is not run-ready", errors)

        payload["artifacts"] = {
            "manifest_sha256": "a" * 64,
            "model_revision": "b" * 40,
        }
        self.assertEqual(
            validate_experiment_config(payload, "fixture", run_ready=True), []
        )

    def test_repository_is_prereg_valid_but_e2_is_not_run_ready_with_placeholders(self) -> None:
        payload = repository_config("e2_healthbench.json")
        self.assertEqual(validate_experiment_config(payload, "fixture"), [])
        errors = "\n".join(
            validate_experiment_config(payload, "fixture", run_ready=True)
        )
        self.assertIn("teacher_template_sha256 is not run-ready", errors)
        self.assertIn("frozen_manifest_sha256 is not run-ready", errors)
        self.assertIn("manifest_sha256 is not run-ready", errors)
        self.assertIn("model_revision_and_prompt_sha256 is not run-ready", errors)
        self.assertNotIn("GLOBAL_REVISION is not run-ready", errors)
        self.assertNotIn("intervene_global_revision is not run-ready", errors)
        self.assertNotIn("repair_global_revision is not run-ready", errors)


if __name__ == "__main__":
    unittest.main()
