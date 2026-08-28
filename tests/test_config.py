import json
import math
import tempfile
import unittest
from pathlib import Path

from rvi_opd.config import validate_config_paths, validate_experiment_config


def valid_config():
    return {
        "id": "D-test",
        "question": "Does the contract hold?",
        "split": "synthetic_only",
        "arms": ["repair", "intervene"],
        "primary_endpoints": ["score"],
        "audit_invariants": ["same_seed"],
    }


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


if __name__ == "__main__":
    unittest.main()
