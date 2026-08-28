import tempfile
import unittest
from pathlib import Path

from rvi_opd.smoke import run_smoke


class SmokeTests(unittest.TestCase):
    def test_smoke_writes_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = run_smoke(output)
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue(report["budget"]["matched"])
            self.assertEqual(
                report["gate"]["rolled_back"]["effective_action"], "repair"
            )
            self.assertIn("not scientific evidence", report["warning"])
            self.assertEqual(
                report["frozen_scale_sha256"],
                report["thresholds"]["scale_artifact_sha256"],
            )
            self.assertIn("optimizer_steps", report["budget"]["match_on"])
            self.assertIn("d0_three_way_interaction_fixture", report)


if __name__ == "__main__":
    unittest.main()
