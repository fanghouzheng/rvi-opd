from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .budget import BudgetLedger, LedgerEntry, audit_match
from .config import load_config, validate_config_paths, validate_upstreams_lock
from .data_audit import build_prompt_manifest
from .execution import (
    EXECUTION_TARGETS,
    execution_readiness,
    validate_execution_policy_path,
)
from .io import atomic_write_json, read_jsonl, write_jsonl
from .models import CostVector, RawStateSignal
from .router import (
    PRIMARY_ROUTER_QUANTILE,
    RouterPolicy,
    ThresholdArtifact,
    calibrate_thresholds,
    route_batch,
)
from .signals import apply_frozen_scale, fit_frozen_scale
from .smoke import run_smoke


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rvi-opd", description="Auditable RvI-OPD experiment utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the deterministic CPU contract test")
    smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))

    validate = subparsers.add_parser("validate-config", help="validate experiment JSON files")
    validate.add_argument("--config-dir", type=Path, default=Path("configs"))
    validate.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="upstreams.lock.json to cross-check model/data revisions (default: sibling of config-dir)",
    )
    validate.add_argument(
        "--run-ready",
        action="store_true",
        help="also reject unresolved SHA256/revision placeholders required before a real run",
    )

    route = subparsers.add_parser("route-jsonl", help="calibrate and route raw state-signal JSONL")
    threshold_source = route.add_mutually_exclusive_group(required=True)
    threshold_source.add_argument("--calibration", type=Path)
    threshold_source.add_argument("--threshold-artifact", type=Path)
    route.add_argument("--input", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)
    route.add_argument("--threshold-output", type=Path)
    route.add_argument("--s1-quantile", type=float, default=PRIMARY_ROUTER_QUANTILE)
    route.add_argument("--s2-quantile", type=float, default=PRIMARY_ROUTER_QUANTILE)
    route.add_argument("--normalization-low", type=float)
    route.add_argument("--normalization-high", type=float)
    route.add_argument("--repetition-threshold", type=float)
    route.add_argument("--paced-zero-rescue", action="store_true")
    route.add_argument("--teacher-revision", default="")
    route.add_argument("--student-revision", default="")
    route.add_argument("--tokenizer-sha256", default="")
    route.add_argument("--vocabulary-sha256", default="")
    route.add_argument("--code-revision", default="")
    route.add_argument("--calibration-split-sha256", default="")
    route.add_argument(
        "--lexicon-artifact-sha256",
        default="",
        help="legacy single-lexicon hash; retained for replay compatibility only",
    )
    route.add_argument(
        "--trd-epistemic-lexicon-artifact-sha256",
        default="",
        help="hash of the 16-phrase TRD onset lexicon artifact",
    )
    route.add_argument(
        "--relay-single-token-lexicon-artifact-sha256",
        default="",
        help="hash of the strict single-token Relay lexicon artifact",
    )
    route.add_argument(
        "--reflection-token-ids",
        default="",
        help="legacy comma-separated reflection IDs; cannot make an artifact production-ready",
    )
    route.add_argument(
        "--trd-epistemic-token-ids",
        default="",
        help="comma-separated tokenizer-specific TRD onset IDs",
    )
    route.add_argument(
        "--relay-single-token-ids",
        default="",
        help="comma-separated strict single-token Relay trigger IDs",
    )
    route.add_argument(
        "--require-production-metadata",
        action="store_true",
        help="reject floating/incomplete threshold artifacts",
    )

    audit = subparsers.add_parser("audit-budget", help="compare two arms in a ledger JSONL")
    audit.add_argument("--ledger", type=Path, required=True)
    audit.add_argument("--left-arm", required=True)
    audit.add_argument("--right-arm", required=True)
    audit.add_argument("--match-on", required=True, help="comma-separated CostVector fields")
    audit.add_argument("--relative-tolerance", type=float, default=0.01)

    prompts = subparsers.add_parser(
        "audit-prompts", help="create a content-free normalized exact-dedup manifest"
    )
    prompts.add_argument("--input", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--id-field", default="id")
    prompts.add_argument("--prompt-field", default="prompt")

    readiness = subparsers.add_parser(
        "execution-readiness",
        help="enforce the HealthBench-first release gate for an execution target",
    )
    readiness.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/execution/healthbench-first.json"),
    )
    readiness.add_argument("--target", choices=sorted(EXECUTION_TARGETS), required=True)
    readiness.add_argument(
        "--gate-result",
        type=Path,
        default=None,
        help="append-only HealthBench gate evidence artifact; required for math targets",
    )
    readiness.add_argument("--output", type=Path, default=None)

    validate_execution = subparsers.add_parser(
        "validate-execution-policy",
        help="validate the frozen HealthBench-first execution-order policy",
    )
    validate_execution.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/execution/healthbench-first.json"),
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_token_ids(value: str, field_name: str) -> List[int]:
    if not value.strip():
        return []
    try:
        token_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated integers") from exc
    if not token_ids or any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"{field_name} must contain non-negative integers")
    return token_ids


def _assert_no_path_collisions(read_paths: List[Path], write_paths: List[Path]) -> None:
    resolved_reads = {path.resolve() for path in read_paths}
    resolved_writes = [path.resolve() for path in write_paths]
    if len(set(resolved_writes)) != len(resolved_writes):
        raise ValueError("route output and threshold output must be different paths")
    overlap = resolved_reads.intersection(resolved_writes)
    if overlap:
        raise ValueError(
            "route outputs must not overwrite inputs: "
            + ", ".join(sorted(str(path) for path in overlap))
        )


def _raw_records(path: Path) -> List[RawStateSignal]:
    records = []
    for row in read_jsonl(path):
        try:
            records.append(RawStateSignal(**row))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid RawStateSignal row: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _ledger(path: Path) -> BudgetLedger:
    ledger = BudgetLedger()
    for row in read_jsonl(path):
        try:
            cost = CostVector(**row["cost"])
            ledger.add(
                LedgerEntry(
                    run_id=row["run_id"],
                    arm=row["arm"],
                    state_id=row["state_id"],
                    requested_action=row["requested_action"],
                    effective_action=row["effective_action"],
                    cost=cost,
                    stratum=row["stratum"],
                    seed=row["seed"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid ledger row: {exc}") from exc
    return ledger


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "smoke":
        report = run_smoke(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-config":
        paths = sorted(args.config_dir.glob("*.json"))
        if not paths:
            print(f"no JSON configs found in {args.config_dir}", file=sys.stderr)
            return 2
        errors = validate_config_paths(paths, run_ready=args.run_ready)
        lock_path = args.lock or args.config_dir.parent / "upstreams.lock.json"
        if lock_path.is_file():
            payloads = []
            for path in paths:
                try:
                    payloads.append(load_config(path))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
            errors.extend(validate_upstreams_lock(lock_path, payloads))
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        mode = "run-ready" if args.run_ready else "preregistration"
        print(f"validated {len(paths)} experiment configs ({mode} mode)")
        return 0
    if args.command == "execution-readiness":
        if args.output is not None:
            protected_inputs = {args.policy.resolve()}
            if args.gate_result is not None:
                protected_inputs.add(args.gate_result.resolve())
            if args.output.resolve() in protected_inputs:
                print(
                    "execution policy error: readiness output must not overwrite "
                    "the policy or gate evidence",
                    file=sys.stderr,
                )
                return 2
        try:
            report = execution_readiness(
                args.policy,
                args.target,
                args.gate_result,
                enforce_clean_checkout=True,
            )
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            print(f"execution policy error: {exc}", file=sys.stderr)
            return 2
        if args.output is not None:
            atomic_write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["order_allowed"] else 1
    if args.command == "validate-execution-policy":
        try:
            errors = validate_execution_policy_path(args.policy)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"execution policy error: {exc}", file=sys.stderr)
            return 2
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"validated execution policy: {args.policy}")
        return 0
    if args.command == "route-jsonl":
        if args.threshold_artifact:
            thresholds = ThresholdArtifact.from_dict(load_config(args.threshold_artifact))
            scale = thresholds.frozen_scale()
            normalization_low = thresholds.normalization_low
            normalization_high = thresholds.normalization_high
            if (
                args.normalization_low is not None
                and args.normalization_low != normalization_low
            ) or (
                args.normalization_high is not None
                and args.normalization_high != normalization_high
            ):
                raise ValueError(
                    "normalization quantiles cannot override a frozen threshold artifact"
                )
        else:
            normalization_low = (
                0.05 if args.normalization_low is None else args.normalization_low
            )
            normalization_high = (
                0.95 if args.normalization_high is None else args.normalization_high
            )
            calibration_raw = _raw_records(args.calibration)
            scale = fit_frozen_scale(
                calibration_raw,
                normalization_low,
                normalization_high,
                vocabulary_sha256=args.vocabulary_sha256,
                code_revision=args.code_revision,
            )
            calibration = apply_frozen_scale(calibration_raw, scale)
            thresholds = calibrate_thresholds(
                calibration,
                s1_quantile=args.s1_quantile,
                s2_quantile=args.s2_quantile,
                teacher_revision=args.teacher_revision,
                student_revision=args.student_revision,
                tokenizer_sha256=args.tokenizer_sha256,
                calibration_split_sha256=(
                    args.calibration_split_sha256 or _sha256_file(args.calibration)
                ),
                normalization_low=normalization_low,
                normalization_high=normalization_high,
                lexicon_artifact_sha256=args.lexicon_artifact_sha256,
                trd_epistemic_lexicon_artifact_sha256=args.trd_epistemic_lexicon_artifact_sha256,
                relay_single_token_lexicon_artifact_sha256=args.relay_single_token_lexicon_artifact_sha256,
                reflection_token_ids=_parse_token_ids(
                    args.reflection_token_ids, "reflection-token-ids"
                ),
                trd_epistemic_token_ids=_parse_token_ids(
                    args.trd_epistemic_token_ids, "trd-epistemic-token-ids"
                ),
                relay_single_token_ids=_parse_token_ids(
                    args.relay_single_token_ids, "relay-single-token-ids"
                ),
                frozen_scale=scale,
            )
        records = apply_frozen_scale(_raw_records(args.input), scale)
        if args.require_production_metadata:
            thresholds.assert_production_ready()
        policy = RouterPolicy(
            repetition_threshold=args.repetition_threshold,
            paced_zero_rescue=args.paced_zero_rescue,
        )
        decisions = route_batch(records, thresholds, policy)
        threshold_output = None
        if args.threshold_output or not args.threshold_artifact:
            threshold_output = args.threshold_output or args.output.with_suffix(
                ".thresholds.json"
            )
        read_paths = [args.input]
        read_paths.append(args.threshold_artifact or args.calibration)
        write_paths = [args.output]
        if threshold_output is not None:
            write_paths.append(threshold_output)
        _assert_no_path_collisions(read_paths, write_paths)
        write_jsonl(
            args.output,
            (
                {
                    **asdict(decision),
                    "action": decision.action.value,
                    "requested_action": decision.action.value,
                    "decision_stage": "requested_pre_gate",
                    "threshold_artifact_sha256": thresholds.artifact_sha256,
                }
                for decision in decisions
            ),
        )
        if threshold_output is not None:
            atomic_write_json(threshold_output, thresholds.to_dict())
        print(f"routed {len(decisions)} states to {args.output}")
        return 0
    if args.command == "audit-budget":
        if not math.isfinite(args.relative_tolerance) or args.relative_tolerance < 0:
            raise ValueError("relative tolerance must be a non-negative finite number")
        ledger = _ledger(args.ledger)
        available_arms = set(ledger.arms())
        missing_arms = {args.left_arm, args.right_arm} - available_arms
        if missing_arms:
            raise ValueError(
                "ledger does not contain arm(s): " + ", ".join(sorted(missing_arms))
            )
        if args.left_arm == args.right_arm:
            raise ValueError("left and right budget arms must differ")
        left = ledger.total(args.left_arm)
        right = ledger.total(args.right_arm)
        match_on = list(
            dict.fromkeys(field.strip() for field in args.match_on.split(",") if field.strip())
        )
        if not match_on:
            raise ValueError("match-on must contain at least one CostVector field")
        unknown_fields = set(match_on) - set(CostVector().to_dict())
        if unknown_fields:
            raise ValueError(
                "unknown cost fields: " + ", ".join(sorted(unknown_fields))
            )
        left_groups = ledger.grouped_totals(args.left_arm)
        right_groups = ledger.grouped_totals(args.right_arm)
        if set(left_groups) != set(right_groups):
            missing_left = sorted(set(right_groups) - set(left_groups))
            missing_right = sorted(set(left_groups) - set(right_groups))
            raise ValueError(
                f"budget arms have different (run_id,stratum,seed) groups; "
                f"missing_left={missing_left}, missing_right={missing_right}"
            )
        group_results = []
        mismatches = []
        for key in sorted(left_groups):
            group_left = left_groups[key]
            group_right = right_groups[key]
            group_tolerances = {
                field: max(
                    abs(float(getattr(group_left, field))),
                    abs(float(getattr(group_right, field))),
                    1.0,
                )
                * args.relative_tolerance
                for field in match_on
            }
            group_mismatches = audit_match(
                group_left, group_right, match_on, group_tolerances
            )
            mismatches.extend(group_mismatches)
            group_results.append(
                {
                    "run_id": key[0],
                    "stratum": key[1],
                    "seed": key[2],
                    "matched": not group_mismatches,
                    "mismatches": [asdict(item) for item in group_mismatches],
                }
            )
        payload = {
            "left_arm": args.left_arm,
            "right_arm": args.right_arm,
            "match_on": match_on,
            "relative_tolerance": args.relative_tolerance,
            "matched": not mismatches,
            "left": left.to_dict(),
            "right": right.to_dict(),
            "mismatches": [asdict(item) for item in mismatches],
            "groups": group_results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not mismatches else 1
    if args.command == "audit-prompts":
        manifest = build_prompt_manifest(
            read_jsonl(args.input), id_field=args.id_field, prompt_field=args.prompt_field
        )
        atomic_write_json(args.output, manifest)
        summary = {
            key: manifest[key]
            for key in (
                "physical_rows",
                "unique_normalized_prompts",
                "duplicate_physical_rows",
                "manifest_sha256",
            )
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 2
