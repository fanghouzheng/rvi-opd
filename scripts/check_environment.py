#!/usr/bin/env python3
"""Validate the active interpreter against the repository environment contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping
from urllib.parse import unquote, urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPOSITORY_ROOT / "environment-lock.json"


def _load_lock() -> Mapping[str, object]:
    with LOCK_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("environment-lock.json must use schema_version 1")
    return payload


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"unsupported numeric version {value!r}")
    return tuple(int(part) for part in parts)


def _check_python(profile: str, contract: Mapping[str, object], errors: List[str]) -> None:
    actual = platform.python_version()
    if profile in {"cpu", "dev"}:
        version = sys.version_info[:2]
        if version < (3, 9) or version >= (3, 13):
            errors.append(f"Python >=3.9,<3.13 required; found {actual}")
        return

    expected = contract.get("python_exact")
    if not isinstance(expected, str):
        errors.append("GPU contract is missing python_exact")
    elif _version_tuple(actual) != _version_tuple(expected):
        errors.append(f"Python {expected} required; found {actual}")


def _check_packages(
    packages: Mapping[str, object], errors: List[str]
) -> Dict[str, str]:
    installed: Dict[str, str] = {}
    for distribution, expected in packages.items():
        if not isinstance(distribution, str) or not isinstance(expected, str):
            errors.append("package pins must map distribution names to version strings")
            continue
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {distribution}=={expected}")
            continue
        installed[distribution] = actual
        if actual != expected:
            errors.append(f"{distribution}: expected {expected}, found {actual}")
    return installed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _parse_requirements_lock(path: Path) -> Dict[str, str]:
    packages: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.count("==") != 1:
            raise ValueError(
                f"unsupported requirement at {path}:{line_number}: {value!r}"
            )
        distribution, version = value.split("==", 1)
        if not distribution or not version or distribution in packages:
            raise ValueError(
                f"invalid or duplicate requirement at {path}:{line_number}: {value!r}"
            )
        packages[distribution] = version
    if not packages:
        raise ValueError(f"empty dependency lock: {path}")
    return packages


def _check_relay_checkout(
    contract: Mapping[str, object], errors: List[str]
) -> Dict[str, object]:
    relay = contract.get("relay")
    if not isinstance(relay, dict):
        errors.append("GPU contract is missing relay metadata")
        return {}

    relay_dir = Path(
        os.environ.get("RVI_RELAY_DIR", REPOSITORY_ROOT / "third_party" / "Relay-OPD")
    ).resolve()
    report: Dict[str, object] = {"path": str(relay_dir)}
    if not (relay_dir / ".git").is_dir():
        errors.append(f"Relay checkout is missing or not Git: {relay_dir}")
        return report

    try:
        commit = _git_output(relay_dir, "rev-parse", "HEAD")
        dirty = _git_output(
            relay_dir, "status", "--porcelain", "--untracked-files=all"
        )
        remote = _git_output(relay_dir, "remote", "get-url", "origin")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        errors.append(f"cannot audit Relay checkout: {exc}")
        return report

    report.update({"commit": commit, "clean": not bool(dirty), "origin": remote})
    trusted_checkout = True
    expected_commit = relay.get("commit")
    if commit != expected_commit:
        errors.append(f"Relay commit: expected {expected_commit}, found {commit}")
        trusted_checkout = False
    if dirty:
        errors.append("Relay checkout has tracked or untracked changes")
        trusted_checkout = False
    allowed_remotes = {
        relay.get("repository"),
        "git@github.com:ZJU-REAL/Relay-OPD.git",
    }
    if remote not in allowed_remotes:
        errors.append(f"unexpected Relay origin: {remote}")
        trusted_checkout = False

    # Never execute code from a checkout whose provenance or content is already
    # known to be wrong. The remaining verifier is trusted only after these
    # fail-closed gates and the lock digest below have passed.
    if not trusted_checkout:
        return report

    relative_lock = relay.get("requirements_lock_path")
    expected_lock_sha = relay.get("requirements_lock_sha256")
    if not isinstance(relative_lock, str) or not isinstance(expected_lock_sha, str):
        errors.append("GPU contract has invalid Relay lock metadata")
        return report
    requirements_lock = relay_dir / relative_lock
    if not requirements_lock.is_file():
        errors.append(f"Relay requirements lock is missing: {requirements_lock}")
        return report
    actual_lock_sha = _sha256(requirements_lock)
    report["requirements_lock_sha256"] = actual_lock_sha
    if actual_lock_sha != expected_lock_sha:
        errors.append(
            f"Relay requirements lock SHA256: expected {expected_lock_sha}, "
            f"found {actual_lock_sha}"
        )
        return report

    try:
        full_lock = _parse_requirements_lock(requirements_lock)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return report
    _check_packages(full_lock, errors)
    report["locked_distributions_checked"] = len(full_lock)
    expected_count = relay.get("locked_distribution_count")
    if len(full_lock) != expected_count:
        errors.append(
            f"Relay lock distribution count: expected {expected_count}, "
            f"found {len(full_lock)}"
        )

    expected_source = (relay_dir / "relay-opd").resolve()
    try:
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        errors.append("Relay editable distribution 'verl' is not installed")
    else:
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            errors.append("installed verl has no direct_url.json editable-source record")
        else:
            try:
                direct_url = json.loads(direct_url_text)
                parsed = urlparse(direct_url.get("url", ""))
                source = Path(unquote(parsed.path)).resolve()
                editable = direct_url.get("dir_info", {}).get("editable") is True
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"invalid verl direct_url.json: {exc}")
            else:
                report["installed_verl_source"] = str(source)
                report["installed_verl_editable"] = editable
                if parsed.scheme != "file" or source != expected_source or not editable:
                    errors.append(
                        "installed verl must be an editable install from the pinned "
                        f"Relay subproject {expected_source}; found {source}"
                    )

    verifier = expected_source / "environment" / "verify_install.py"
    if not verifier.is_file():
        errors.append(f"Relay verifier is missing: {verifier}")
        return report
    verifier_environment = os.environ.copy()
    verifier_environment.update(
        {"RELAY_OPD_STRICT_VERSIONS": "1", "RELAY_OPD_REQUIRE_CUDA": "1"}
    )
    try:
        verification = subprocess.run(
            [sys.executable, str(verifier)],
            cwd=str(expected_source),
            env=verifier_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"Relay verify_install.py could not run: {exc}")
    else:
        report["upstream_verifier_passed"] = verification.returncode == 0
        if verification.returncode != 0:
            detail = (verification.stderr or verification.stdout).strip()
            errors.append(f"Relay verify_install.py failed: {detail[-2000:]}")
    return report


def _check_gpu_system(errors: List[str]) -> Dict[str, str]:
    observed = {
        "platform": platform.system(),
        "architecture": platform.machine(),
    }
    if platform.system() != "Linux":
        errors.append(f"GPU profile requires Linux; found {platform.system()}")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        errors.append(f"GPU profile requires x86_64; found {platform.machine()}")

    try:
        import torch
    except (ImportError, OSError) as exc:
        errors.append(f"cannot import torch for CUDA verification: {exc}")
    else:
        observed["torch_cuda"] = str(torch.version.cuda)
        observed["cuda_available"] = str(torch.cuda.is_available()).lower()
        if not str(torch.version.cuda).startswith("13.0"):
            errors.append(f"PyTorch CUDA 13.0 build required; found {torch.version.cuda}")
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")

    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        errors.append(f"nvidia-smi is unavailable: {exc}")
    else:
        if query.returncode != 0 or not query.stdout.strip():
            errors.append("nvidia-smi could not query an NVIDIA driver/GPU")
        else:
            observed["nvidia_driver"] = query.stdout.splitlines()[0].strip()
    return observed


def check_environment(profile: str) -> Dict[str, object]:
    lock = _load_lock()
    contract_key = "gpu_relay_cu130" if profile == "gpu" else profile
    contract = lock.get(contract_key)
    if not isinstance(contract, dict):
        raise ValueError(f"missing environment contract: {contract_key}")

    errors: List[str] = []
    _check_python(profile, contract, errors)
    if profile == "dev":
        bootstrap = lock.get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise ValueError("missing environment contract: bootstrap")
        bootstrap_packages = bootstrap.get("packages", {})
        dev_packages = contract.get("packages", {})
        if not isinstance(bootstrap_packages, dict) or not isinstance(
            dev_packages, dict
        ):
            raise ValueError("bootstrap/dev packages must be objects")
        packages = {**bootstrap_packages, **dev_packages}
    elif profile == "gpu":
        packages = contract.get("key_packages", {})
    else:
        packages = {}
    if not isinstance(packages, dict):
        raise ValueError(f"{contract_key}.packages must be an object")

    report: Dict[str, object] = {
        "profile": profile,
        "python": platform.python_version(),
        "packages": _check_packages(packages, errors),
    }
    if profile == "gpu":
        report["relay"] = _check_relay_checkout(contract, errors)
        report["system"] = _check_gpu_system(errors)
    report["ok"] = not errors
    report["errors"] = errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "dev", "gpu"), required=True)
    arguments = parser.parse_args()
    try:
        report = check_environment(arguments.profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"environment contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
