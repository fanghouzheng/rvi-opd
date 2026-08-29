import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from rvi_opd.cli import _ledger, _raw_records, main


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]

    def test_route_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "routes.jsonl"
            status = main(
                [
                    "route-jsonl",
                    "--calibration",
                    str(self.repo / "examples/calibration_states.jsonl"),
                    "--input",
                    str(self.repo / "examples/states_to_route.jsonl"),
                    "--output",
                    str(output),
                    "--teacher-revision",
                    "a" * 40,
                    "--student-revision",
                    "b" * 40,
                    "--tokenizer-sha256",
                    "c" * 64,
                    "--vocabulary-sha256",
                    "f" * 64,
                    "--code-revision",
                    "1" * 40,
                    "--trd-epistemic-lexicon-artifact-sha256",
                    "d" * 64,
                    "--relay-single-token-lexicon-artifact-sha256",
                    "e" * 64,
                    "--trd-epistemic-token-ids",
                    "20,10,20",
                    "--relay-single-token-ids",
                    "40,30,40",
                    "--require-production-metadata",
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".thresholds.json").is_file())
            first_row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(len(first_row["threshold_artifact_sha256"]), 64)
            self.assertEqual(first_row["decision_stage"], "requested_pre_gate")
            self.assertEqual(first_row["requested_action"], first_row["action"])
            self.assertNotIn("effective_action", first_row)
            threshold_payload = json.loads(
                output.with_suffix(".thresholds.json").read_text(encoding="utf-8")
            )
            self.assertEqual(threshold_payload["normalization_low"], 0.05)
            self.assertEqual(threshold_payload["normalization_high"], 0.95)
            self.assertEqual(len(threshold_payload["scale_artifact_sha256"]), 64)
            self.assertAlmostEqual(threshold_payload["divergence_q_low"], 0.11)
            self.assertAlmostEqual(threshold_payload["divergence_q_high"], 1.28)
            self.assertAlmostEqual(threshold_payload["compatibility_q_low"], 0.17)
            self.assertAlmostEqual(threshold_payload["compatibility_q_high"], 0.90)
            self.assertEqual(threshold_payload["vocabulary_sha256"], "f" * 64)
            self.assertEqual(threshold_payload["code_revision"], "1" * 40)
            self.assertTrue(threshold_payload["production_ready"])
            self.assertEqual(threshold_payload["signal_schema_version"], "rvi-signals-v3")
            self.assertEqual(threshold_payload["trd_epistemic_token_ids"], [10, 20])
            self.assertEqual(threshold_payload["relay_single_token_ids"], [30, 40])
            self.assertEqual(threshold_payload["s1_quantile"], 0.8)
            self.assertEqual(threshold_payload["s2_quantile"], 0.8)
            calibration_bytes = (self.repo / "examples/calibration_states.jsonl").read_bytes()
            self.assertEqual(
                threshold_payload["calibration_split_sha256"],
                hashlib.sha256(calibration_bytes).hexdigest(),
            )

            replay = Path(temporary) / "replay.jsonl"
            replay_status = main(
                [
                    "route-jsonl",
                    "--threshold-artifact",
                    str(output.with_suffix(".thresholds.json")),
                    "--input",
                    str(self.repo / "examples/states_to_route.jsonl"),
                    "--output",
                    str(replay),
                    "--require-production-metadata",
                ]
            )
            self.assertEqual(replay_status, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), replay.read_text(encoding="utf-8"))

    def test_validate_config_exposes_preregistration_and_run_ready_modes(self) -> None:
        self.assertEqual(
            main(["validate-config", "--config-dir", str(self.repo / "configs")]),
            0,
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(
                [
                    "validate-config",
                    "--config-dir",
                    str(self.repo / "configs"),
                    "--run-ready",
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(stderr.getvalue().count("is not run-ready"), 7)

    def test_route_replay_is_independent_of_batch_id_and_peers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            thresholds = temporary_path / "thresholds.json"
            main(
                [
                    "route-jsonl",
                    "--calibration",
                    str(self.repo / "examples/calibration_states.jsonl"),
                    "--input",
                    str(self.repo / "examples/states_to_route.jsonl"),
                    "--output",
                    str(temporary_path / "initial.jsonl"),
                    "--threshold-output",
                    str(thresholds),
                    "--normalization-low",
                    "0.0",
                    "--normalization-high",
                    "1.0",
                ]
            )
            target = {
                "state_id": "target",
                "problem_id": "p",
                "trajectory_id": "t",
                "token_index": 3,
                "divergence": 0.8,
                "compatibility": 0.8,
                "s2": 0.003,
                "batch_id": "batch-a",
            }
            first_rows = [
                target,
                {
                    **target,
                    "state_id": "peer-low",
                    "divergence": 0.0,
                    "compatibility": 0.0,
                },
            ]
            second_rows = [
                {**target, "batch_id": "batch-z"},
                {
                    **target,
                    "state_id": "peer-high",
                    "divergence": 100.0,
                    "compatibility": 1.0,
                    "batch_id": "batch-z",
                },
            ]
            inputs = []
            outputs = []
            for suffix, rows in (("first", first_rows), ("second", second_rows)):
                input_path = temporary_path / f"{suffix}.jsonl"
                output_path = temporary_path / f"{suffix}-routes.jsonl"
                input_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                inputs.append(input_path)
                outputs.append(output_path)
                main(
                    [
                        "route-jsonl",
                        "--threshold-artifact",
                        str(thresholds),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )
            first = json.loads(outputs[0].read_text(encoding="utf-8").splitlines()[0])
            second = json.loads(outputs[1].read_text(encoding="utf-8").splitlines()[0])
            self.assertNotEqual(first["batch_id"], second["batch_id"])
            for field in (
                "state_id",
                "action",
                "reason",
                "s1",
                "s2",
                "threshold_artifact_sha256",
            ):
                self.assertEqual(first[field], second[field])

    def test_route_rejects_output_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "same.json"
            with self.assertRaises(ValueError):
                main(
                    [
                        "route-jsonl",
                        "--calibration",
                        str(self.repo / "examples/calibration_states.jsonl"),
                        "--input",
                        str(self.repo / "examples/states_to_route.jsonl"),
                        "--output",
                        str(output),
                        "--threshold-output",
                        str(output),
                    ]
                )

    def test_invalid_rows_include_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "bad-states.jsonl"
            state_path.write_text(
                json.dumps(
                    {
                        "state_id": "s",
                        "problem_id": "p",
                        "trajectory_id": "t",
                        "token_index": 0,
                        "divergence": 0.1,
                        "compatibility": 2.0,
                        "s2": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as state_error:
                _raw_records(state_path)
            self.assertIn(str(state_path), str(state_error.exception))

            ledger_path = Path(temporary) / "bad-ledger.jsonl"
            ledger_path.write_text(
                json.dumps(
                    {
                        "run_id": "r",
                        "arm": "repair",
                        "state_id": "s",
                        "requested_action": "repair",
                        "effective_action": "repair",
                        "stratum": "x",
                        "seed": 1,
                        "cost": {"teacher_scored_tokens": -1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ledger_error:
                _ledger(ledger_path)
            self.assertIn(str(ledger_path), str(ledger_error.exception))

    def test_budget_audit(self) -> None:
        status = main(
            [
                "audit-budget",
                "--ledger",
                str(self.repo / "examples/budget_ledger.jsonl"),
                "--left-arm",
                "repair",
                "--right-arm",
                "intervene",
                    "--match-on",
                    (
                        "examples,teacher_scored_tokens,student_supervised_tokens,"
                        "optimizer_steps"
                    ),
            ]
        )
        self.assertEqual(status, 0)

    def test_budget_audit_rejects_missing_arm_and_empty_axes(self) -> None:
        ledger = str(self.repo / "examples/budget_ledger.jsonl")
        with self.assertRaises(ValueError):
            main(
                [
                    "audit-budget",
                    "--ledger",
                    ledger,
                    "--left-arm",
                    "missing",
                    "--right-arm",
                    "repair",
                    "--match-on",
                    "examples",
                ]
            )

    def test_budget_audit_rejects_nonfinite_tolerance_and_unknown_axis(self) -> None:
        ledger = str(self.repo / "examples/budget_ledger.jsonl")
        for extra_args in (
            ["--match-on", "examples", "--relative-tolerance", "nan"],
            ["--match-on", "examples", "--relative-tolerance", "inf"],
            ["--match-on", "not_a_cost_field"],
        ):
            with self.assertRaises(ValueError):
                main(
                    [
                        "audit-budget",
                        "--ledger",
                        ledger,
                        "--left-arm",
                        "repair",
                        "--right-arm",
                        "intervene",
                        *extra_args,
                    ]
                )
        with self.assertRaises(ValueError):
            main(
                [
                    "audit-budget",
                    "--ledger",
                    ledger,
                    "--left-arm",
                    "repair",
                    "--right-arm",
                    "intervene",
                    "--match-on",
                    ",",
                ]
            )


if __name__ == "__main__":
    unittest.main()
