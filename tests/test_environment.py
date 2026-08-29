import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_environment_checker(root: Path):
    path = root / "scripts" / "check_environment.py"
    spec = importlib.util.spec_from_file_location("rvi_environment_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load environment checker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnvironmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.environment = json.loads(
            (cls.root / "environment-lock.json").read_text(encoding="utf-8")
        )
        cls.upstreams = json.loads(
            (cls.root / "upstreams.lock.json").read_text(encoding="utf-8")
        )

    def test_cpu_and_dev_dependency_files_match_machine_contract(self) -> None:
        runtime_lines = [
            line.strip()
            for line in (self.root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(runtime_lines, [])

        def requirement_lines(filename: str) -> set[str]:
            return {
                line.strip()
                for line in (self.root / filename)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }

        bootstrap_expected = {
            f"{name}=={version}"
            for name, version in self.environment["bootstrap"]["packages"].items()
        }
        dev_expected = {
            f"{name}=={version}"
            for name, version in self.environment["dev"]["packages"].items()
        }
        self.assertEqual(
            requirement_lines("requirements-bootstrap.txt"), bootstrap_expected
        )
        self.assertEqual(requirement_lines("requirements-dev.txt"), dev_expected)
        self.assertEqual(
            self.environment["cpu"]["third_party_runtime_packages"], []
        )

    def test_gpu_lock_and_upstream_roles_are_consistent(self) -> None:
        gpu = self.environment["gpu_relay_cu130"]
        relay = self.upstreams["code"]["relay_opd"]
        verl = self.upstreams["code"]["verl"]
        self.assertEqual(gpu["relay"]["commit"], relay["commit"])
        self.assertEqual(
            gpu["relay"]["requirements_lock_sha256"],
            relay["environment_lock_sha256"],
        )
        self.assertEqual(
            gpu["relay"]["locked_distribution_count"],
            relay["locked_distribution_count"],
        )
        self.assertEqual(
            self.environment["standalone_verl"]["commit"], verl["commit"]
        )
        self.assertTrue(verl["incompatible_with_relay_environment"])

        requirements = (self.root / "requirements-gpu-cu130.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn(relay["commit"], requirements)
        self.assertIn(relay["environment_lock_sha256"], requirements)
        self.assertIn("create_gpu_env.sh", requirements)

        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.9,<3.13"', pyproject)
        self.assertIn('requires = ["setuptools==80.10.2"]', pyproject)

    def test_cpu_environment_checker_runs_without_optional_packages(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(self.root / "scripts" / "check_environment.py"),
                "--profile",
                "cpu",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["profile"], "cpu")

    def test_gpu_checker_never_executes_untrusted_relay_checkout(self) -> None:
        checker = load_environment_checker(self.root)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkout = Path(temporary_directory) / "Relay-OPD"
            environment = checkout / "relay-opd" / "environment"
            environment.mkdir(parents=True)
            requirements_lock = environment / "requirements.lock.txt"
            requirements_lock.write_text("example-package==1.0\n", encoding="utf-8")
            executed_marker = checkout / "verifier-executed"
            (environment / "verify_install.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(executed_marker)!r}).write_text('unsafe')\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.email", "ci@example.test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.name", "CI"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/ZJU-REAL/Relay-OPD.git",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "add", "."], check=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "commit", "-m", "fixture"],
                check=True,
                capture_output=True,
            )
            actual_commit = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            actual_hash = hashlib.sha256(requirements_lock.read_bytes()).hexdigest()
            base_contract = {
                "relay": {
                    "repository": "https://github.com/ZJU-REAL/Relay-OPD.git",
                    "commit": actual_commit,
                    "requirements_lock_path": "relay-opd/environment/requirements.lock.txt",
                    "requirements_lock_sha256": actual_hash,
                    "locked_distribution_count": 1,
                }
            }

            with mock.patch.dict(os.environ, {"RVI_RELAY_DIR": str(checkout)}):
                wrong_commit = json.loads(json.dumps(base_contract))
                wrong_commit["relay"]["commit"] = "0" * 40
                errors = []
                checker._check_relay_checkout(wrong_commit, errors)
                self.assertTrue(any("Relay commit" in error for error in errors))
                self.assertFalse(executed_marker.exists())

                wrong_lock = json.loads(json.dumps(base_contract))
                wrong_lock["relay"]["requirements_lock_sha256"] = "0" * 64
                errors = []
                checker._check_relay_checkout(wrong_lock, errors)
                self.assertTrue(any("SHA256" in error for error in errors))
                self.assertFalse(executed_marker.exists())


if __name__ == "__main__":
    unittest.main()
