from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .stats import ci_is_equivalent


POLICY_SCHEMA_VERSION = 1
GATE_RESULT_SCHEMA = "rvi-healthbench-math-release-v1"
POLICY_ID = "healthbench-first-v1"
MATRIX_REVISION = "2026-08-30"
EXECUTION_SCOPE = "execution_order_only"
EXECUTION_SCOPE_NOTE = (
    "This check enforces HealthBench-before-Math ordering only; it does not replace "
    "validate-config --run-ready, C0/data checks, or GPU/adapter backend readiness."
)
EXECUTION_TARGETS = {
    "E2:C0",
    "E2:W0",
    "E2:resource_pilot",
    "D1:medical",
    "D2:medical",
    "E2:core_train",
    "E2:healthbench_final",
    "D1:math",
    "D2:math",
    "D0",
    "D3",
    "D4",
    "D5",
    "E1",
    "A1-A8:math",
}
PRE_GATE_TARGETS = {
    "E2:C0",
    "E2:W0",
    "E2:resource_pilot",
    "D1:medical",
    "D2:medical",
    "E2:core_train",
    "E2:healthbench_final",
}
MATH_TARGETS = EXECUTION_TARGETS - PRE_GATE_TARGETS
CORE_TRAINING_SEEDS = [13, 17, 23]
CORE_TRAINED_ARMS = {
    "vanilla_opd",
    "relay_opd",
    "trd_canonical_full_vocab",
    "repair_only",
    "intervene_only",
    "rvi_opd",
    "a2_action_shuffled",
}
NON_ORACLE_BASELINES = {
    "vanilla_opd",
    "relay_opd",
    "trd_canonical_full_vocab",
}
SINGLE_ACTION_BASELINES = {"repair_only", "intervene_only"}
REMAINING_E2_ROWS = {
    "sft",
    "fastopd_relay_fixed_prefix_reproduction",
    "skd",
    "ta_opd",
    "tip_select",
}
MATH_CONFIGS = {
    "configs/ablations.json",
    "configs/d0_factorial.json",
    "configs/d1_signal_calibration.json",
    "configs/d2_paired_continuation.json",
    "configs/d3_detached.json",
    "configs/d4_degenerate_prefix.json",
    "configs/d5_paced_rescue.json",
    "configs/e1_math.json",
}
REQUIRED_TRUE_FLAGS = {
    "c0_passed",
    "w0_passed",
    "grader_repeatability_passed",
    "all_required_runs_complete",
    "all_required_artifact_hashes_verified",
    "settings_frozen_before_healthbench_outputs",
    "math_configs_frozen_before_healthbench_outputs",
    "healthbench_never_used_for_training_or_hyperparameter_selection",
    "hard_reused_full_completions",
    "all_failures_and_retries_retained",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_bundle_sha256(repository_root: Path, paths: Sequence[str]) -> str:
    root = repository_root.resolve()
    digest = hashlib.sha256()
    for relative in sorted(paths):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"config path escapes repository root: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"bound config is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _unknown_or_missing(
    value: Any, field: str, expected_keys: Set[str], errors: List[str]
) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return False
    missing = expected_keys - set(value)
    unknown = set(value) - expected_keys
    if missing:
        errors.append(f"{field} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        errors.append(f"{field} contains unknown fields: {', '.join(sorted(unknown))}")
    return not missing and not unknown


def _string_set(value: Any, field: str, errors: List[str]) -> Set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field} must be a list of strings")
        return set()
    if len(value) != len(set(value)):
        errors.append(f"{field} must not contain duplicates")
    return set(value)


def validate_execution_policy(
    policy: Mapping[str, Any], repository_root: Path
) -> List[str]:
    errors: List[str] = []
    top_keys = {
        "schema_version",
        "policy_id",
        "matrix_revision",
        "branch",
        "amendment_type",
        "scientific_outcomes_seen_before_amendment",
        "producer_target",
        "allowed_pre_gate_targets",
        "math_targets",
        "failure_status",
        "gate_result_schema",
        "config_bindings",
        "healthbench_core",
        "required_true_flags",
        "decision_rule",
    }
    _unknown_or_missing(policy, "execution policy", top_keys, errors)
    fixed = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "matrix_revision": MATRIX_REVISION,
        "branch": "healthbench-first",
        "amendment_type": "prospective_execution_order_and_stop_rule",
        "scientific_outcomes_seen_before_amendment": False,
        "producer_target": "E2:healthbench_final",
        "failure_status": "NOT_RUN_HEALTHBENCH_GATE",
        "gate_result_schema": GATE_RESULT_SCHEMA,
    }
    for field, expected in fixed.items():
        if policy.get(field) != expected:
            errors.append(f"execution policy.{field} must equal {expected!r}")

    pre_gate = _string_set(
        policy.get("allowed_pre_gate_targets"),
        "execution policy.allowed_pre_gate_targets",
        errors,
    )
    math_targets = _string_set(
        policy.get("math_targets"), "execution policy.math_targets", errors
    )
    if pre_gate != PRE_GATE_TARGETS:
        errors.append("execution policy pre-gate target set is not the frozen set")
    if math_targets != MATH_TARGETS:
        errors.append("execution policy math target set is not the frozen set")
    if pre_gate & math_targets or pre_gate | math_targets != EXECUTION_TARGETS:
        errors.append("execution policy must cover every target exactly once")
    if policy.get("producer_target") not in pre_gate:
        errors.append("gate producer must be allowed before its own gate")

    bindings = policy.get("config_bindings")
    if _unknown_or_missing(
        bindings, "execution policy.config_bindings", {"e2_config", "math_configs"}, errors
    ):
        assert isinstance(bindings, dict)
        if bindings.get("e2_config") != "configs/e2_healthbench.json":
            errors.append("execution policy must bind configs/e2_healthbench.json")
        math_configs = _string_set(
            bindings.get("math_configs"),
            "execution policy.config_bindings.math_configs",
            errors,
        )
        if math_configs != MATH_CONFIGS:
            errors.append("execution policy math config bundle is incomplete or changed")
        for relative in [bindings.get("e2_config"), *sorted(math_configs)]:
            if not isinstance(relative, str):
                continue
            config_path = repository_root / relative
            if not config_path.is_file():
                errors.append(f"execution policy bound config is missing: {relative}")
                continue
            try:
                bound_config = load_json_object(config_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"execution policy bound config is invalid: {relative}: {exc}")
                continue
            if bound_config.get("matrix_revision") != MATRIX_REVISION:
                errors.append(
                    f"execution policy bound config {relative} must use "
                    f"matrix_revision {MATRIX_REVISION}"
                )

    core = policy.get("healthbench_core")
    core_keys = {
        "training_seeds",
        "trained_arms",
        "evaluation_sampling_seeds",
        "required_training_run_count",
        "required_evaluation_manifest_count",
        "full_prompt_count",
        "hard_prompt_count",
        "hard_reuses_full_completions",
        "remaining_e2_rows_after_gate",
    }
    if _unknown_or_missing(core, "execution policy.healthbench_core", core_keys, errors):
        assert isinstance(core, dict)
        if core.get("training_seeds") != CORE_TRAINING_SEEDS:
            errors.append("HealthBench core training seeds must be [13, 17, 23]")
        trained_arms = _string_set(
            core.get("trained_arms"), "execution policy.healthbench_core.trained_arms", errors
        )
        if trained_arms != CORE_TRAINED_ARMS:
            errors.append("HealthBench core trained arm set is not frozen correctly")
        expected_evaluation = {
            arm: CORE_TRAINING_SEEDS for arm in sorted(CORE_TRAINED_ARMS)
        }
        expected_evaluation.update(
            {"base": CORE_TRAINING_SEEDS, "teacher_upper_bound": [13]}
        )
        if core.get("evaluation_sampling_seeds") != expected_evaluation:
            errors.append("HealthBench evaluation sampling policy is not frozen correctly")
        fixed_core = {
            "required_training_run_count": 21,
            "required_evaluation_manifest_count": 25,
            "full_prompt_count": 5000,
            "hard_prompt_count": 1000,
            "hard_reuses_full_completions": True,
        }
        for field, expected in fixed_core.items():
            if core.get(field) != expected:
                errors.append(f"HealthBench core {field} must equal {expected!r}")
        remaining = _string_set(
            core.get("remaining_e2_rows_after_gate"),
            "execution policy.healthbench_core.remaining_e2_rows_after_gate",
            errors,
        )
        if remaining != REMAINING_E2_ROWS:
            errors.append("remaining E2 row set is not frozen correctly")

    flags = _string_set(
        policy.get("required_true_flags"), "execution policy.required_true_flags", errors
    )
    if flags != REQUIRED_TRUE_FLAGS:
        errors.append("execution policy required flag set is incomplete or changed")

    rule = policy.get("decision_rule")
    rule_keys = {
        "type",
        "material_margin",
        "equivalence_margin",
        "safety_noninferiority_margin",
        "base_comparator",
        "non_oracle_baselines",
        "single_action_baselines",
        "a2_comparator",
        "rubric_mixed_excluded",
        "hard_is_secondary_only",
        "decision_on_pass",
        "decision_on_fail",
    }
    if _unknown_or_missing(rule, "execution policy.decision_rule", rule_keys, errors):
        assert isinstance(rule, dict)
        fixed_rule = {
            "type": "intersection_union_all_required",
            "material_margin": 0.01,
            "equivalence_margin": 0.01,
            "safety_noninferiority_margin": 0.01,
            "base_comparator": "base",
            "a2_comparator": "a2_action_shuffled",
            "rubric_mixed_excluded": True,
            "hard_is_secondary_only": True,
            "decision_on_pass": "GO_MATH",
            "decision_on_fail": "STOP_AFTER_HEALTHBENCH",
        }
        for field, expected in fixed_rule.items():
            if rule.get(field) != expected:
                errors.append(f"execution policy decision_rule.{field} must equal {expected!r}")
        if _string_set(
            rule.get("non_oracle_baselines"),
            "execution policy.decision_rule.non_oracle_baselines",
            errors,
        ) != NON_ORACLE_BASELINES:
            errors.append("non-oracle baseline gate set is not frozen correctly")
        if _string_set(
            rule.get("single_action_baselines"),
            "execution policy.decision_rule.single_action_baselines",
            errors,
        ) != SINGLE_ACTION_BASELINES:
            errors.append("single-action baseline gate set is not frozen correctly")
    return errors


def _repository_root_for_policy(policy_path: Path) -> Path:
    resolved = policy_path.resolve()
    expected_tail = ("configs", "execution", "healthbench-first.json")
    if tuple(resolved.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError(
            "execution policy must be the canonical "
            "configs/execution/healthbench-first.json file"
        )
    return resolved.parents[2]


def validate_execution_policy_path(policy_path: Path) -> List[str]:
    policy_path = policy_path.resolve()
    repository_root = _repository_root_for_policy(policy_path)
    policy = load_json_object(policy_path)
    return validate_execution_policy(policy, repository_root)


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _interval(value: Any, field: str, lower_field: str) -> Tuple[float, float, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    expected = {"estimate", lower_field, "upper_95ci"}
    if set(value) != expected:
        raise ValueError(f"{field} must contain exactly {sorted(expected)}")
    estimate = _finite_number(value["estimate"], f"{field}.estimate")
    lower = _finite_number(value[lower_field], f"{field}.{lower_field}")
    upper = _finite_number(value["upper_95ci"], f"{field}.upper_95ci")
    if not lower <= estimate <= upper:
        raise ValueError(f"{field} interval must contain its estimate")
    return estimate, lower, upper


def _git_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or not COMMIT_RE.fullmatch(result.stdout.strip()):
        raise ValueError("cannot resolve the repository Git revision")
    return result.stdout.strip()


def _assert_clean_checkout(repository_root: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError("cannot audit repository working tree")
    if result.stdout.strip():
        raise ValueError("execution readiness requires a clean repository checkout")


def _run_pairs(
    rows: Any, field: str, *, evaluation: bool = False
) -> Tuple[Set[Tuple[str, int]], bool]:
    if not isinstance(rows, list):
        raise ValueError(f"{field} must be a list")
    pairs: Set[Tuple[str, int]] = set()
    valid_hashes = True
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        if evaluation:
            expected = {
                "arm",
                "sampling_seed",
                "completion_manifest_sha256",
                "grader_manifest_sha256",
            }
            seed_field = "sampling_seed"
            hash_fields = ("completion_manifest_sha256", "grader_manifest_sha256")
        else:
            expected = {"arm", "seed", "run_manifest_sha256", "checkpoint_sha256"}
            seed_field = "seed"
            hash_fields = ("run_manifest_sha256", "checkpoint_sha256")
        if set(row) != expected:
            raise ValueError(f"{field}[{index}] must contain exactly {sorted(expected)}")
        arm = row.get("arm")
        seed = row.get(seed_field)
        if not isinstance(arm, str) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{field}[{index}] has an invalid arm or seed")
        pair = (arm, seed)
        if pair in pairs:
            raise ValueError(f"{field} contains duplicate run {pair}")
        pairs.add(pair)
        for hash_field in hash_fields:
            if not isinstance(row.get(hash_field), str) or not SHA256_RE.fullmatch(
                row[hash_field]
            ):
                valid_hashes = False
    return pairs, valid_hashes


def evaluate_healthbench_gate(
    policy_path: Path,
    gate_result_path: Path,
    *,
    enforce_clean_checkout: bool = True,
) -> Dict[str, Any]:
    policy_path = policy_path.resolve()
    repository_root = _repository_root_for_policy(policy_path)
    policy = load_json_object(policy_path)
    policy_errors = validate_execution_policy(policy, repository_root)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))
    if enforce_clean_checkout:
        _assert_clean_checkout(repository_root)

    result = load_json_object(gate_result_path)
    result_keys = {
        "schema",
        "policy_sha256",
        "code_revision",
        "e2_config_sha256",
        "math_config_bundle_sha256",
        "flags",
        "training_runs",
        "evaluation_manifests",
        "counts",
        "metrics",
    }
    if set(result) != result_keys:
        raise ValueError(f"gate result must contain exactly {sorted(result_keys)}")
    if result.get("schema") != GATE_RESULT_SCHEMA:
        raise ValueError(f"gate result schema must equal {GATE_RESULT_SCHEMA!r}")
    if result.get("policy_sha256") != sha256_file(policy_path):
        raise ValueError("gate result policy SHA256 does not match the active policy")
    code_revision = result.get("code_revision")
    if not isinstance(code_revision, str) or not COMMIT_RE.fullmatch(code_revision):
        raise ValueError("gate result code_revision must be a 40-hex Git commit")
    if code_revision != _git_head(repository_root):
        raise ValueError("gate result code_revision does not match the checkout")

    bindings = policy["config_bindings"]
    e2_path = repository_root / bindings["e2_config"]
    if result.get("e2_config_sha256") != sha256_file(e2_path):
        raise ValueError("gate result E2 config SHA256 does not match the checkout")
    math_bundle = config_bundle_sha256(repository_root, bindings["math_configs"])
    if result.get("math_config_bundle_sha256") != math_bundle:
        raise ValueError("gate result math config bundle SHA256 does not match the checkout")

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, requirement: str, observed: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "requirement": requirement,
                "observed": observed,
            }
        )

    flags = result.get("flags")
    if not isinstance(flags, dict) or set(flags) != REQUIRED_TRUE_FLAGS:
        raise ValueError("gate result flags must exactly match the policy required flags")
    for name in sorted(REQUIRED_TRUE_FLAGS):
        if type(flags[name]) is not bool:
            raise ValueError(f"gate result flags.{name} must be a boolean")
        add_check(f"flag:{name}", flags[name], "true", flags[name])

    core = policy["healthbench_core"]
    training_pairs, training_hashes_valid = _run_pairs(
        result.get("training_runs"), "gate result training_runs"
    )
    expected_training = {
        (arm, seed) for arm in CORE_TRAINED_ARMS for seed in CORE_TRAINING_SEEDS
    }
    add_check(
        "complete_training_run_set",
        training_pairs == expected_training,
        f"exactly {len(expected_training)} frozen arm/seed pairs",
        len(training_pairs),
    )
    add_check(
        "training_artifact_hashes",
        training_hashes_valid,
        "every run/checkpoint hash is 64-hex",
        training_hashes_valid,
    )

    evaluation_pairs, evaluation_hashes_valid = _run_pairs(
        result.get("evaluation_manifests"),
        "gate result evaluation_manifests",
        evaluation=True,
    )
    expected_evaluations = {
        (arm, seed)
        for arm, seeds in core["evaluation_sampling_seeds"].items()
        for seed in seeds
    }
    add_check(
        "complete_evaluation_manifest_set",
        evaluation_pairs == expected_evaluations,
        f"exactly {len(expected_evaluations)} frozen arm/sampling-seed pairs",
        len(evaluation_pairs),
    )
    add_check(
        "evaluation_artifact_hashes",
        evaluation_hashes_valid,
        "every completion/grader hash is 64-hex",
        evaluation_hashes_valid,
    )

    counts = result.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "full_prompt_count",
        "hard_prompt_count",
    }:
        raise ValueError("gate result counts must contain Full and Hard prompt counts")
    for field in ("full_prompt_count", "hard_prompt_count"):
        if type(counts[field]) is not int:
            raise ValueError(f"gate result counts.{field} must be an integer")
    add_check(
        "full_prompt_count",
        counts["full_prompt_count"] == core["full_prompt_count"],
        str(core["full_prompt_count"]),
        counts["full_prompt_count"],
    )
    add_check(
        "hard_prompt_count",
        counts["hard_prompt_count"] == core["hard_prompt_count"],
        str(core["hard_prompt_count"]),
        counts["hard_prompt_count"],
    )

    metrics = result.get("metrics")
    metric_keys = {
        "rvi_vs_base",
        "rvi_vs_non_oracle",
        "rvi_vs_single_actions",
        "rvi_vs_a2",
        "rubric_mechanism",
        "negative_violation_noninferiority",
    }
    if not isinstance(metrics, dict) or set(metrics) != metric_keys:
        raise ValueError(f"gate result metrics must contain exactly {sorted(metric_keys)}")
    rule = policy["decision_rule"]
    material_margin = float(rule["material_margin"])

    base_estimate, base_lower, _ = _interval(
        metrics["rvi_vs_base"],
        "metrics.rvi_vs_base",
        "lower_95ci",
    )
    add_check(
        "rvi_beats_base",
        base_lower > 0,
        "paired two-level bootstrap lower 95% CI > 0",
        base_lower,
    )
    add_check(
        "rvi_material_gain_over_base",
        base_estimate >= material_margin,
        f"point estimate >= {material_margin}",
        base_estimate,
    )

    non_oracle = metrics["rvi_vs_non_oracle"]
    if not isinstance(non_oracle, dict) or set(non_oracle) != NON_ORACLE_BASELINES:
        raise ValueError("rvi_vs_non_oracle must contain every frozen baseline")
    non_oracle_estimates: Dict[str, float] = {}
    non_oracle_lowers: Dict[str, float] = {}
    for comparator in sorted(NON_ORACLE_BASELINES):
        estimate, lower, _ = _interval(
            non_oracle[comparator],
            f"metrics.rvi_vs_non_oracle.{comparator}",
            "simultaneous_lower_95ci",
        )
        non_oracle_estimates[comparator] = estimate
        non_oracle_lowers[comparator] = lower
    strongest_non_oracle = min(non_oracle_estimates, key=non_oracle_estimates.get)
    add_check(
        "rvi_beats_each_non_oracle",
        all(value > 0 for value in non_oracle_lowers.values()),
        "every simultaneous lower 95% CI > 0",
        non_oracle_lowers,
    )
    add_check(
        "rvi_material_gain_over_strongest_non_oracle",
        non_oracle_estimates[strongest_non_oracle] >= material_margin,
        f"point estimate >= {material_margin}",
        {
            "comparator": strongest_non_oracle,
            "estimate": non_oracle_estimates[strongest_non_oracle],
        },
    )

    single_actions = metrics["rvi_vs_single_actions"]
    if not isinstance(single_actions, dict) or set(single_actions) != SINGLE_ACTION_BASELINES:
        raise ValueError("rvi_vs_single_actions must contain repair_only and intervene_only")
    single_estimates: Dict[str, float] = {}
    single_lowers: Dict[str, float] = {}
    for comparator in sorted(SINGLE_ACTION_BASELINES):
        estimate, lower, _ = _interval(
            single_actions[comparator],
            f"metrics.rvi_vs_single_actions.{comparator}",
            "simultaneous_lower_95ci",
        )
        single_estimates[comparator] = estimate
        single_lowers[comparator] = lower
    strongest_single = min(single_estimates, key=single_estimates.get)
    add_check(
        "rvi_beats_each_single_action",
        all(value > 0 for value in single_lowers.values()),
        "every simultaneous lower 95% CI > 0",
        single_lowers,
    )
    add_check(
        "rvi_material_gain_over_strongest_single_action",
        single_estimates[strongest_single] >= material_margin,
        f"point estimate >= {material_margin}",
        {"comparator": strongest_single, "estimate": single_estimates[strongest_single]},
    )

    a2 = metrics["rvi_vs_a2"]
    if not isinstance(a2, dict) or set(a2) != {
        "estimate",
        "lower_95ci",
        "upper_95ci",
        "leave_one_seed_out_estimates",
    }:
        raise ValueError("rvi_vs_a2 has an invalid schema")
    _, a2_lower, _ = _interval(
        {key: a2[key] for key in ("estimate", "lower_95ci", "upper_95ci")},
        "metrics.rvi_vs_a2",
        "lower_95ci",
    )
    loo = a2["leave_one_seed_out_estimates"]
    if not isinstance(loo, list) or len(loo) != len(CORE_TRAINING_SEEDS):
        raise ValueError("rvi_vs_a2 leave-one-seed-out list must contain three estimates")
    loo_values = [
        _finite_number(value, f"metrics.rvi_vs_a2.leave_one_seed_out_estimates[{index}]")
        for index, value in enumerate(loo)
    ]
    add_check("rvi_beats_a2", a2_lower > 0, "lower 95% CI > 0", a2_lower)
    add_check(
        "a2_leave_one_seed_out_stability",
        all(value > 0 for value in loo_values),
        "all three leave-one-seed-out estimates > 0",
        loo_values,
    )

    rubric = metrics["rubric_mechanism"]
    rubric_keys = {
        "did_holm_adjusted_lower_95ci",
        "repair_insertable_lower_95ci",
        "intervene_insertable_lower_95ci",
        "intervene_global_revision_lower_95ci",
        "repair_global_revision_90ci",
    }
    if not isinstance(rubric, dict) or set(rubric) != rubric_keys:
        raise ValueError("rubric_mechanism has an invalid schema")
    directional = {
        field: _finite_number(rubric[field], f"metrics.rubric_mechanism.{field}")
        for field in rubric_keys - {"repair_global_revision_90ci"}
    }
    add_check(
        "rubric_directional_and_did",
        all(value > 0 for value in directional.values()),
        "Holm-adjusted DiD and every directional lower CI > 0",
        directional,
    )
    repair_ci = rubric["repair_global_revision_90ci"]
    if not isinstance(repair_ci, dict) or set(repair_ci) != {"lower", "upper"}:
        raise ValueError("repair_global_revision_90ci must contain lower and upper")
    repair_lower = _finite_number(repair_ci["lower"], "repair_global_revision_90ci.lower")
    repair_upper = _finite_number(repair_ci["upper"], "repair_global_revision_90ci.upper")
    if repair_lower > repair_upper:
        raise ValueError("repair_global_revision_90ci lower exceeds upper")
    equivalence_margin = float(rule["equivalence_margin"])
    add_check(
        "repair_global_revision_tost",
        ci_is_equivalent(repair_lower, repair_upper, equivalence_margin),
        f"90% CI strictly inside (-{equivalence_margin}, +{equivalence_margin})",
        {"lower": repair_lower, "upper": repair_upper},
    )

    safety = metrics["negative_violation_noninferiority"]
    expected_safety = {"base", *NON_ORACLE_BASELINES}
    if not isinstance(safety, dict) or set(safety) != expected_safety:
        raise ValueError("negative_violation_noninferiority comparator set is invalid")
    safety_upper: Dict[str, float] = {}
    for comparator, value in safety.items():
        field = f"metrics.negative_violation_noninferiority.{comparator}"
        if not isinstance(value, dict) or set(value) != {"upper_95ci"}:
            raise ValueError(f"{field} must contain exactly ['upper_95ci']")
        safety_upper[comparator] = _finite_number(
            value["upper_95ci"], f"{field}.upper_95ci"
        )
    safety_margin = float(rule["safety_noninferiority_margin"])
    safety_targets = {"base", strongest_non_oracle}
    add_check(
        "negative_violation_safety_veto",
        all(safety_upper[name] <= safety_margin for name in safety_targets),
        f"upper 95% CI <= {safety_margin} versus Base and strongest non-oracle",
        {name: safety_upper[name] for name in sorted(safety_targets)},
    )

    passed = all(check["passed"] for check in checks)
    decision = rule["decision_on_pass"] if passed else rule["decision_on_fail"]
    return {
        "schema": GATE_RESULT_SCHEMA,
        "scope": EXECUTION_SCOPE,
        "scope_note": EXECUTION_SCOPE_NOTE,
        "policy_id": policy["policy_id"],
        "decision": decision,
        "launch_math": passed,
        "failure_status": None if passed else policy["failure_status"],
        "gate_result_sha256": sha256_file(gate_result_path),
        "bindings": {
            "policy_sha256": result["policy_sha256"],
            "code_revision": result["code_revision"],
            "e2_config_sha256": result["e2_config_sha256"],
            "math_config_bundle_sha256": result["math_config_bundle_sha256"],
        },
        "checks": checks,
    }


def execution_readiness(
    policy_path: Path,
    target: str,
    gate_result_path: Optional[Path] = None,
    *,
    enforce_clean_checkout: bool = True,
) -> Dict[str, Any]:
    policy_path = policy_path.resolve()
    repository_root = _repository_root_for_policy(policy_path)
    policy = load_json_object(policy_path)
    errors = validate_execution_policy(policy, repository_root)
    if errors:
        raise ValueError("; ".join(errors))
    if target not in EXECUTION_TARGETS:
        raise ValueError(f"unknown execution target: {target}")
    if enforce_clean_checkout:
        _assert_clean_checkout(repository_root)
    active_bindings = {
        "policy_sha256": sha256_file(policy_path),
        "code_revision": _git_head(repository_root),
        "e2_config_sha256": sha256_file(
            repository_root / policy["config_bindings"]["e2_config"]
        ),
        "math_config_bundle_sha256": config_bundle_sha256(
            repository_root, policy["config_bindings"]["math_configs"]
        ),
    }
    base_report = {
        "policy_id": policy["policy_id"],
        "target": target,
        "scope": EXECUTION_SCOPE,
        "scope_note": EXECUTION_SCOPE_NOTE,
        "run_readiness_assessed": False,
        "bindings": active_bindings,
    }
    if target in PRE_GATE_TARGETS:
        return {
            **base_report,
            "order_allowed": True,
            "status": "ORDER_ALLOWED_PRE_GATE",
            "gate_required": False,
        }
    if gate_result_path is None or not gate_result_path.is_file():
        return {
            **base_report,
            "order_allowed": False,
            "status": "ORDER_BLOCKED_PENDING_HEALTHBENCH_GATE",
            "gate_required": True,
        }
    gate = evaluate_healthbench_gate(
        policy_path,
        gate_result_path,
        enforce_clean_checkout=False,
    )
    return {
        **base_report,
        "order_allowed": gate["launch_math"],
        "status": (
            "ORDER_ALLOWED_MATH_AFTER_HEALTHBENCH"
            if gate["launch_math"]
            else policy["failure_status"]
        ),
        "gate_required": True,
        "gate": gate,
    }
