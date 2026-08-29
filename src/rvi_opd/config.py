from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


REQUIRED_EXPERIMENT_FIELDS = {
    "id",
    "question",
    "split",
    "arms",
    "primary_endpoints",
    "audit_invariants",
    "matrix_revision",
    "citation_cutoff",
    "concurrent_policy",
}

REQUIRED_STRING_FIELDS = (
    "id",
    "question",
    "split",
    "matrix_revision",
    "citation_cutoff",
    "concurrent_policy",
)
REQUIRED_STRING_LIST_FIELDS = ("arms", "primary_endpoints", "audit_invariants")

# All machine-readable experiment files in this repository are snapshots of the
# same preregistration matrix.  Keeping the revision and policy in every file
# makes it impossible to accidentally run a stale D0/E1/E2 config alongside a
# newer one.
MATRIX_REVISION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MATRIX_CONCURRENT_POLICY = "post_cutoff_treat_as_concurrent_do_not_cite_or_compare"
ROUTER_PRIMARY_QUANTILES = {"s1": 0.80, "s2": 0.80}

PROBABILITY_FIELD_NAMES = {
    "acceptance_rate",
    "error_rate",
    "p",
    "p_hat",
    "pass_rate",
    "probability",
    "probabilities",
    "quantile",
    "quantiles",
    "clip_ratio",
    "relative_tolerance",
    "repetition_rate",
    "state_bands",
    "success_rate",
    "top_p",
    "trigger_rate",
}
PROBABILITY_FIELD_SUFFIXES = (
    "_fraction",
    "_fractions",
    "_probability",
    "_probabilities",
    "_quantile",
    "_quantiles",
    "_acceptance_rate",
    "_error_rate",
    "_pass_rate",
    "_repetition_rate",
    "_success_rate",
    "_trigger_rate",
    "_top_p",
)

# A boolean is an ``int`` subclass in Python.  These names identify fields for
# which accepting ``true``/``false`` would silently turn a count or seed into
# 1/0.  Experiment-specific fixed contracts below cover the remaining numeric
# fields whose names are less regular.
INTEGER_FIELD_NAMES = {
    "benchmark_count",
    "completions_per_checkpoint_prompt",
    "effective_global_batch_size",
    "eos_reserve",
    "full_prompt_count",
    "global_batch_size",
    "hard_prompt_count",
    "k_signal",
    "max_input_tokens",
    "max_model_length",
    "max_prompt_tokens",
    "max_response_tokens",
    "minimum_distinct_prompts_per_cell",
    "minimum_distinct_prompts_per_signal_action_s2_subgroup",
    "minimum_distinct_prompts_per_signal_action_stratum",
    "minimum_prompts",
    "minimum_states",
    "minimum_states_per_cell",
    "minimum_states_per_signal_action_s2_subgroup",
    "minimum_states_per_signal_action_stratum",
    "n_repeats_primary",
    "optimizer_steps",
    "pad_token_id",
    "paired_continuation_seeds_per_state",
    "pilot_prompt_count",
    "ppo_epochs",
    "ppo_minibatch_size",
    "questions",
    "randomization_seed",
    "rollout_top_k",
    "rollouts_per_prompt",
    "rows",
    "sample_seed",
    "shots",
    "stop_token_ids",
    "teacher_tokenizer_model_max_length",
    "student_tokenizer_model_max_length",
    "top_k",
    "training_epochs",
    "training_seeds",
    "validated_pretraining_context",
}
INTEGER_FIELD_SUFFIXES = (
    "_count",
    "_epochs",
    "_repeats",
    "_seed",
    "_token_id",
    "_token_ids",
)
NUMERIC_FIELD_NAMES = {
    "clip_ratio",
    "learning_rate",
    "rollout_temperature",
    "source_weight",
    "temperature",
}

# Every checked-in experiment has an ID-specific top-level contract.  The
# generic base schema remains available to callers defining their own IDs, but
# a known ID cannot degrade into a superficially valid config after a section
# is deleted or renamed.
ID_REQUIRED_FIELD_TYPES = {
    "A1-A8": {
        "arm_contracts": dict,
        "A2_assignment": dict,
        "parameter_grids": dict,
        "efficiency_metrics": list,
        "protected_core": list,
        "cut_first_order": list,
        "important_equivalence": str,
        "multiple_testing": dict,
    },
    "D0": {
        "design": dict,
        "repair": dict,
        "intervene": dict,
        "gate": str,
        "budget": dict,
        "secondary_endpoints": list,
        "confirmatory_contrast": dict,
        "subgroup_predictions": dict,
        "success_gate": dict,
        "analysis": dict,
    },
    "D1": {
        "signals": dict,
        "success_gate": dict,
        "acceptance_gate_calibration": dict,
        "freeze_artifact": list,
        "artifact_schema": dict,
    },
    "D2": {
        "sample": dict,
        "repair_probe": dict,
        "continuation_protocol": dict,
        "equivalence_margin": dict,
        "success_gate": dict,
    },
    "D3": {
        "probe_scope": dict,
        "detached_contract": dict,
        "action_event_schema": dict,
        "eos_stop_token_ids": list,
        "gate": str,
        "budget": dict,
        "success_gate": dict,
    },
    "D4": {
        "sample": dict,
        "bypass_signals": list,
        "signal_definitions": dict,
        "challenge_selection": dict,
        "teacher_alternative_direction": dict,
        "route": str,
        "assignment": dict,
        "status": str,
    },
    "D5": {
        "selection": dict,
        "selection_artifact": dict,
        "sample_size": dict,
        "paced_estimators": dict,
        "estimators": dict,
        "one_cycle_schedule": dict,
        "action_contracts": dict,
        "neighbor_transfer": dict,
        "success_gate": dict,
    },
    "E1": {
        "models": dict,
        "data": dict,
        "training": dict,
        "routing_contract": dict,
        "baseline_contracts": dict,
        "main_table_arms": list,
        "supplementary_arms": list,
        "seed_policy": dict,
        "evaluation": dict,
        "success_gate": dict,
    },
    "E2": {
        "models": dict,
        "training_data": dict,
        "denylist": list,
        "training": dict,
        "routing_contract": dict,
        "main_table_arms": list,
        "supplementary_arms": list,
        "baseline_contracts": dict,
        "official_evaluation": dict,
        "seed_policy": dict,
        "rubric_annotation": dict,
        "key_secondary_endpoints": list,
        "statistics": dict,
        "success_gate": dict,
    },
}

ABLATION_ARMS = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"}
ABLATION_PROTECTED_CORE = ["D0", "D2", "D3", "E1", "E2", "A1", "A2"]
ABLATION_CUT_FIRST_ORDER = [
    "optional_medical_8B_and_ChatDoctor_robustness",
    "A6",
    "A4_TIP_style_control",
    "D5",
]
A5_BUDGET_REPORTING = [
    "teacher_prefill_tokens",
    "teacher_scored_tokens",
    "gate_teacher_scored_tokens",
    "teacher_generated_tokens",
    "teacher_gpu_seconds",
]
A5_FIXED = {
    "global_signal_quantiles": [0.70, 0.75, 0.80],
    "teacher_gpu_seconds_per_prompt_relative_to_frozen_default": [0.5, 1.0, 2.0],
    "budget_reference": "D1_blinded_compute_pilot_frozen_RvI_teacher_gpu_seconds_per_prompt",
    "budget_enforcement": (
        "hard_cap_per_independent_training_run_finish_current_atomic_minibatch_"
        "record_overrun_and_never_refund_rejected_interventions"
    ),
    "budget_reporting": A5_BUDGET_REPORTING,
    "teacher_query_position_relative_tolerance": [0.005, 0.01],
    "relay_paragraph_lengths": [1, 3, 5],
    "cooldown_valid_triggers": [0, 1, 2],
}

D0_ARMS = {
    "dl_top_repair",
    "dl_top_intervene",
    "di_top_repair",
    "di_top_intervene",
}
D0_DESIGN_KEYS = {
    "factors",
    "signal_cell_rules",
    "s2_analysis_strata",
    "mechanism_probe",
    "policy_training_check",
    "minimum_states",
    "minimum_states_per_cell",
    "minimum_states_per_signal_action_s2_subgroup",
    "minimum_distinct_prompts_per_signal_action_s2_subgroup",
    "minimum_prompts",
    "minimum_distinct_prompts_per_cell",
    "paired_continuation_seeds_per_state",
    "prefix_position",
    "sample_size_rule",
}
D0_S2_STRATA_KEYS = {
    "low",
    "high",
    "role",
    "minimum_states_per_signal_action_stratum",
    "minimum_distinct_prompts_per_signal_action_stratum",
    "subgroup_contrast",
    "confirmatory",
    "multiplicity",
    "claim_boundary",
}
D0_S2_STRATA_FIXED = {
    "low": "s2_at_or_below_frozen_q25",
    "high": "s2_at_or_above_frozen_q75",
    "role": "prespecified_subgroup_and_mechanism_readout_not_a_randomized_factor",
    "minimum_states_per_signal_action_stratum": 64,
    "minimum_distinct_prompts_per_signal_action_stratum": 20,
    "subgroup_contrast": "delta_s2=(mu_I_high-mu_R_high)-(mu_I_low-mu_R_low)",
    "confirmatory": False,
    "multiplicity": "exploratory_same_Holm_family_as_subgroup_endpoints",
    "claim_boundary": "subgroup evidence cannot be called a two-signal routing theorem",
}

D1_ACCEPTANCE_GATE_KEYS = {
    "improvements",
    "statistic",
    "threshold",
    "decision",
    "decision_semantics",
    "probe_rollouts_per_event",
    "probe_aggregation",
    "forbidden",
    "position_role",
}
D1_ARTIFACT_SCHEMA_KEYS = {
    "version",
    "threshold_artifact",
    "joint_gate_artifact",
    "run_manifest_requires_both_artifact_hashes",
    "replay_must_be_read_only",
}
D1_THRESHOLD_ARTIFACT_KEYS = {
    "schema",
    "required_numeric_fields",
    "required_hash_fields",
    "required_token_id_fields",
    "forbidden_field",
}
D1_JOINT_GATE_ARTIFACT_KEYS = {
    "schema",
    "required_fields",
    "probe_rollouts_per_event",
    "aggregation",
    "one_way_binding",
}

E1_ARMS = {
    "base",
    "teacher_upper_bound",
    "sft",
    "kd",
    "vanilla_opd",
    "fastopd_relay_fixed_prefix_reproduction",
    "skd",
    "tip_select",
    "ta_opd",
    "relay_opd",
    "trd_canonical_full_vocab",
    "trd_relay_top128_reproduction",
    "repair_only",
    "intervene_only",
    "detached",
    "rvi_opd",
    "a2_action_shuffled",
    "grpo",
}
E1_MAIN_ARMS = {
    "base",
    "teacher_upper_bound",
    "sft",
    "vanilla_opd",
    "fastopd_relay_fixed_prefix_reproduction",
    "skd",
    "ta_opd",
    "tip_select",
    "relay_opd",
    "trd_canonical_full_vocab",
    "rvi_opd",
}
E1_SUPPLEMENTARY_ARMS = E1_ARMS - E1_MAIN_ARMS
E1_BENCHMARKS_32 = ["AIME2024", "AIME2025", "AIME2026", "AMC2023", "HMMT_Feb2026"]
E1_BENCHMARKS_4 = ["MATH500", "OlympiadBench"]
E1_SEED_POLICY = {
    "core_five_seeds": [13, 17, 23, 29, 31],
    "core_trained_arms": [
        "vanilla_opd",
        "ta_opd",
        "relay_opd",
        "rvi_opd",
        "a2_action_shuffled",
    ],
    "evaluation_only_arms_without_training_seeds": ["base", "teacher_upper_bound"],
    "mechanism_three_seeds": [13, 17, 23],
    "mechanism_arms": [
        "repair_only",
        "intervene_only",
        "detached",
        "trd_canonical_full_vocab",
        "trd_relay_top128_reproduction",
    ],
    "secondary_seeds": [13, 17, 23],
    "secondary_arms": [
        "sft",
        "kd",
        "fastopd_relay_fixed_prefix_reproduction",
        "skd",
        "tip_select",
        "grpo",
    ],
}
E1_TRAINED_ARM_PARTITION = {
    "core": E1_SEED_POLICY["core_trained_arms"],
    "mechanism": E1_SEED_POLICY["mechanism_arms"],
    "secondary": E1_SEED_POLICY["secondary_arms"],
    "evaluation_only": E1_SEED_POLICY["evaluation_only_arms_without_training_seeds"],
}
E1_TRAINING_FIXED = {
    "framework": "verl",
    "max_prompt_tokens": 2048,
    "max_response_tokens": 16384,
    "max_model_length": 34817,
    "rollout_do_sample": True,
    "rollout_temperature": 1.0,
    "rollout_top_p": 1.0,
    "rollout_top_k": 0,
    "rollouts_per_prompt": 1,
    "global_batch_size": 128,
    "ppo_minibatch_size": 128,
    "ppo_epochs": 1,
    "clip_ratio": 0.2,
    "learning_rate": 1e-6,
    "learning_rate_schedule": "constant",
    "training_epochs": 1,
}
E1_EVALUATION_FIXED = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "stop_token_ids": [151645, 151643],
    "pad_token_id": 151643,
    "max_response_tokens": 32768,
    "max_model_length": 34817,
    "same_problem_and_sampling_seed_manifest_across_arms": True,
}
E1_MODEL_KEYS = {
    "teacher",
    "teacher_revision",
    "student",
    "student_revision",
    "student_lineage",
    "prompt_contract",
    "revision_policy",
    "shared_vocab_required",
}
E1_PROMPT_CONTRACT_KEYS = {
    "canonical_serializer",
    "native_template_enable_thinking",
    "canonical_serializer_thinking",
    "system_prompt",
    "teacher_template_sha256",
    "student_template_sha256",
    "identical_native_templates_required",
    "native_template_enable_thinking_is_record_only",
    "canonical_rendered_token_ids_must_match",
    "stop_token_ids",
    "pad_token_id",
    "tokenizer_json_sha256",
    "vocab_json_sha256",
    "teacher_merges_sha256",
    "student_merges_sha256",
    "runtime_context_limit",
    "validated_pretraining_context",
}
E1_EVALUATION_KEYS = {
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "stop_token_ids",
    "pad_token_id",
    "max_response_tokens",
    "max_model_length",
    "metric_definition",
    "same_problem_and_sampling_seed_manifest_across_arms",
    "cross_benchmark_contamination_audit",
    "benchmarks_32_samples",
    "benchmarks_4_samples",
    "benchmark_count",
    "sampling_by_benchmark",
    "macro_aggregation",
    "problem_level_bootstrap",
    "forgetting_check",
}

E2_ARMS = {
    "base",
    "teacher_upper_bound",
    "sft",
    "vanilla_opd",
    "fastopd_relay_fixed_prefix_reproduction",
    "skd",
    "ta_opd",
    "tip_select",
    "relay_opd",
    "repair_only",
    "intervene_only",
    "trd_canonical_full_vocab",
    "rvi_opd",
    "a2_action_shuffled",
}
E2_MAIN_ARMS = {
    "base",
    "teacher_upper_bound",
    "sft",
    "vanilla_opd",
    "fastopd_relay_fixed_prefix_reproduction",
    "skd",
    "ta_opd",
    "tip_select",
    "relay_opd",
    "trd_canonical_full_vocab",
    "rvi_opd",
}
E2_SUPPLEMENTARY_ARMS = E2_ARMS - E2_MAIN_ARMS
E2_TRAINING_FIXED = {
    "framework": "verl_at_upstreams_lock_revision",
    "max_prompt_tokens": 2048,
    "max_response_tokens": 8192,
    "max_model_length": 40960,
    "rollout_do_sample": True,
    "rollout_temperature": 1.0,
    "rollout_top_p": 1.0,
    "rollout_top_k": 0,
    "stop_token_ids": [151645, 151643],
    "pad_token_id": 151643,
    "rollouts_per_prompt": 1,
    "effective_global_batch_size": 128,
    "microbatch_and_gradient_accumulation": (
        "chosen_in_C0_without_changing_effective_global_batch"
    ),
    "learning_rate": 1e-6,
    "learning_rate_schedule": "constant",
    "training_epochs": 1,
}
E2_ANSWER_GENERATION_FIXED = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "stop_token_ids": [151645, 151643],
    "pad_token_id": 151643,
    "max_input_tokens": 36864,
    "max_response_tokens": 4096,
    "max_model_length": 40960,
    "eos_reserve": 0,
    "max_new_tokens_includes_stop_token": True,
    "overflow_policy": "preflight_failure_and_report_never_silent_benchmark_truncation",
    "completions_per_checkpoint_prompt": 1,
    "same_prompt_and_sampling_manifest_across_arms": True,
}
E2_SEED_POLICY = {
    "training_seeds": [13, 17, 23],
    "trained_arms": [
        "sft",
        "vanilla_opd",
        "fastopd_relay_fixed_prefix_reproduction",
        "skd",
        "ta_opd",
        "tip_select",
        "relay_opd",
        "repair_only",
        "intervene_only",
        "trd_canonical_full_vocab",
        "rvi_opd",
        "a2_action_shuffled",
    ],
    "primary_main_trained_arms": [
        "sft",
        "vanilla_opd",
        "fastopd_relay_fixed_prefix_reproduction",
        "skd",
        "ta_opd",
        "tip_select",
        "relay_opd",
        "trd_canonical_full_vocab",
        "rvi_opd",
    ],
    "secondary_arms": ["repair_only", "intervene_only", "a2_action_shuffled"],
    "evaluation_only_arms_without_training_seeds": ["base", "teacher_upper_bound"],
}
E2_TRAINED_ARM_PARTITION = {
    "primary_main": E2_SEED_POLICY["primary_main_trained_arms"],
    "secondary": E2_SEED_POLICY["secondary_arms"],
    "evaluation_only": E2_SEED_POLICY["evaluation_only_arms_without_training_seeds"],
}
E2_MODEL_KEYS = {
    "teacher",
    "teacher_revision",
    "student",
    "student_revision",
    "student_checkpoint_role",
    "student_lineage_base_id",
    "student_posttrained",
    "student_is_instruct_named_checkpoint",
    "optional_robustness_student",
    "optional_robustness_student_revision",
    "revision_policy",
    "chat_template",
    "shared_vocab_required",
    "teacher_must_not_be_evaluator",
    "compatibility_gate",
}
E2_MODEL_FIXED = {
    "teacher": "Qwen/Qwen3-4B-Instruct-2507",
    "teacher_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "student": "Qwen/Qwen3-0.6B",
    "student_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "student_checkpoint_role": "student",
    "student_lineage_base_id": "Qwen/Qwen3-0.6B-Base",
    "student_posttrained": True,
    "student_is_instruct_named_checkpoint": False,
    "shared_vocab_required": True,
    "teacher_must_not_be_evaluator": True,
}
E2_CHAT_TEMPLATE_KEYS = {
    "native_template_enable_thinking",
    "canonical_serializer_thinking",
    "canonical_serializer",
    "system_prompt",
    "teacher_template_sha256",
    "student_template_sha256",
    "rendered_prompt_compatibility",
    "stop_token_ids",
    "pad_token_id",
    "identical_for_teacher_student",
    "identical_within_all_arms",
    "teacher_tokenizer_json_sha256",
    "student_tokenizer_json_sha256",
    "vocabulary_sha256",
    "teacher_merges_sha256",
    "student_merges_sha256",
    "teacher_tokenizer_model_max_length",
    "student_tokenizer_model_max_length",
    "runtime_context_limit_overrides_tokenizer_metadata",
}
E2_OFFICIAL_EVALUATION_KEYS = {
    "implementation",
    "dataset",
    "staged_evaluation",
    "revision",
    "dataset_files",
    "datasets",
    "full_prompt_count",
    "hard_prompt_count",
    "hard_is_subset_of_full",
    "reuse_full_completions_for_hard_subset",
    "score_formula",
    "negative_endpoint_formula",
    "grader",
    "answer_generation",
    "do_not_modify_official_items_or_score",
}
E2_ANSWER_GENERATION_KEYS = set(E2_ANSWER_GENERATION_FIXED)
E2_STAGED_EVALUATION_KEYS = {
    "pilot_prompt_count",
    "pilot_name",
    "pilot_purpose",
    "pilot_is_not_confirmatory",
    "pilot_checkpoint_scope",
    "pilot_prerequisite",
    "pilot_outputs_excluded_from_final_arm_comparisons",
    "final_full_evaluation",
    "hard_uses_indexed_full_completions",
    "final_sampling_manifest_is_distinct_from_pilot",
}
E2_RUBRIC_SAMPLE_KEYS = {
    "prompts",
    "sampling_frame",
    "stratify_by",
    "sample_seed",
    "label_all_rubric_items_for_selected_prompts",
    "manifest_sha256",
}

ROUTING_CONTRACT_KEYS = {
    "calibration",
    "primary_quantiles",
    "analysis_band_quantiles",
    "threshold_scope",
    "trajectory_relative_ranks",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
IMMUTABLE_REVISION_PREFIXES = (
    "teacher_",
    "student_",
    "model_",
    "code_",
    "dataset_",
    "upstream_",
    "optional_robustness_",
)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("top-level config must be an object")
    return payload


def _is_probability_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in PROBABILITY_FIELD_NAMES or normalized.endswith(
        PROBABILITY_FIELD_SUFFIXES
    )


def _is_integer_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in INTEGER_FIELD_NAMES or normalized.endswith(INTEGER_FIELD_SUFFIXES)


def _is_numeric_field(field_name: str) -> bool:
    return field_name.strip().lower() in NUMERIC_FIELD_NAMES


def _validate_value(
    value: Any,
    field_path: str,
    errors: List[str],
    probability_context: bool = False,
    integer_context: bool = False,
    numeric_context: bool = False,
) -> None:
    if value is None:
        errors.append(f"{field_path} must not be null")
        return
    if isinstance(value, bool):
        if probability_context:
            errors.append(f"{field_path} must be numeric and in [0, 1]")
        elif integer_context:
            errors.append(f"{field_path} must be an integer, not a boolean")
        elif numeric_context:
            errors.append(f"{field_path} must be numeric, not a boolean")
        return
    if isinstance(value, str):
        if probability_context:
            errors.append(f"{field_path} must be numeric and in [0, 1]")
        elif integer_context:
            errors.append(f"{field_path} must be an integer")
        elif numeric_context:
            errors.append(f"{field_path} must be numeric")
        elif value == "":
            errors.append(f"{field_path} must be a non-empty string")
        return
    if isinstance(value, int):
        if value < 0:
            errors.append(f"{field_path} must be non-negative")
        elif probability_context and value > 1:
            errors.append(f"{field_path} must be in [0, 1]")
        return
    if isinstance(value, float):
        if integer_context:
            errors.append(f"{field_path} must be an integer")
        elif not math.isfinite(value):
            errors.append(f"{field_path} must be finite")
        elif value < 0:
            errors.append(f"{field_path} must be non-negative")
        elif probability_context and value > 1:
            errors.append(f"{field_path} must be in [0, 1]")
        return
    if isinstance(value, list):
        if not value:
            errors.append(f"{field_path} must be a non-empty list")
            return
        for index, item in enumerate(value):
            _validate_value(
                item,
                f"{field_path}[{index}]",
                errors,
                probability_context,
                integer_context,
                numeric_context,
            )
        return
    if isinstance(value, dict):
        if not value:
            errors.append(f"{field_path} must be a non-empty object")
            return
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{field_path} contains an empty or non-string field name")
                continue
            child_path = f"{field_path}.{key}"
            _validate_value(
                item,
                child_path,
                errors,
                probability_context or _is_probability_field(key),
                integer_context or _is_integer_field(key),
                numeric_context or _is_numeric_field(key),
            )
        return
    errors.append(f"{field_path} has unsupported type {type(value).__name__}")


def _validate_id_required_fields(
    payload: Dict[str, Any], source: str, errors: List[str]
) -> None:
    experiment_id = payload.get("id")
    field_types = ID_REQUIRED_FIELD_TYPES.get(experiment_id)
    if field_types is None:
        return
    missing = set(field_types) - set(payload)
    if missing:
        errors.append(
            f"{source}: {experiment_id} missing experiment fields: "
            f"{', '.join(sorted(missing))}"
        )
    unknown = set(payload) - REQUIRED_EXPERIMENT_FIELDS - set(field_types)
    if unknown:
        errors.append(
            f"{source}: {experiment_id} contains unknown top-level fields: "
            f"{', '.join(sorted(unknown))}"
        )
    for field, expected_type in field_types.items():
        if field not in payload:
            continue
        value = payload[field]
        if type(value) is not expected_type:
            errors.append(
                f"{source}: {experiment_id}.{field} must be a "
                f"{expected_type.__name__}"
            )


def _validate_object_schema(
    value: Any,
    field_path: str,
    required_keys: Set[str],
    allowed_keys: Set[str],
    errors: List[str],
) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{field_path} must be an object")
        return False
    missing = required_keys - set(value)
    if missing:
        errors.append(f"{field_path} missing fields: {', '.join(sorted(missing))}")
    unknown = set(value) - allowed_keys
    if unknown:
        errors.append(f"{field_path} contains unknown fields: {', '.join(sorted(unknown))}")
    return not missing and not unknown


def _validate_fixed_values(
    value: Any,
    field_path: str,
    expected_values: Dict[str, Any],
    errors: List[str],
    allowed_keys: Optional[Set[str]] = None,
    required_keys: Optional[Set[str]] = None,
) -> None:
    allowed = set(expected_values) if allowed_keys is None else allowed_keys
    required = set(expected_values) if required_keys is None else required_keys
    if not _validate_object_schema(value, field_path, required, allowed, errors):
        if not isinstance(value, dict):
            return
    assert isinstance(value, dict)
    for key, expected in expected_values.items():
        if key not in value:
            continue
        actual = value[key]
        if isinstance(expected, bool):
            valid_type = type(actual) is bool
            expected_type = "a boolean"
        elif isinstance(expected, int):
            valid_type = type(actual) is int
            expected_type = "an integer"
        elif isinstance(expected, float):
            valid_type = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            expected_type = "a number"
        else:
            valid_type = type(actual) is type(expected)
            expected_type = {
                str: "a string",
                list: "a list",
                dict: "an object",
            }.get(type(expected), f"a {type(expected).__name__}")
        if not valid_type:
            errors.append(f"{field_path}.{key} must be {expected_type}")
            continue
        if actual != expected:
            errors.append(f"{field_path}.{key} must equal {expected!r}")


def _validate_exact_string_set(
    value: Any, field_path: str, expected: Set[str], errors: List[str]
) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field_path} must be a list of strings")
        return
    actual = set(value)
    if len(actual) != len(value):
        errors.append(f"{field_path} must not contain duplicates")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(
            f"{field_path} must contain the frozen arm set; missing={missing}, extra={extra}"
        )


def _validate_run_ready_values(
    value: Any, field_path: str, errors: List[str], field_name: str = ""
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _validate_run_ready_values(item, f"{field_path}.{key}", errors, key)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_run_ready_values(item, f"{field_path}[{index}]", errors, field_name)
        return
    if not isinstance(value, str):
        return

    normalized = field_name.lower()
    if "sha256" in normalized and not SHA256_RE.fullmatch(value):
        errors.append(
            f"{field_path} is not run-ready: expected a frozen 64-hex SHA-256, "
            f"got {value!r}"
        )
    elif (
        normalized == "revision"
        or (
            normalized.endswith("_revision")
            and normalized.startswith(IMMUTABLE_REVISION_PREFIXES)
        )
    ) and not IMMUTABLE_REVISION_RE.fullmatch(value):
        errors.append(
            f"{field_path} is not run-ready: expected a 40-hex immutable revision, "
            f"got {value!r}"
        )


def _validate_required_string_lists(
    payload: Dict[str, Any], source: str, errors: List[str]
) -> None:
    for field in REQUIRED_STRING_LIST_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, list):
            errors.append(f"{source}: {field} must be a non-empty list")
            continue
        seen = set()
        for index, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(f"{source}: {field}[{index}] must be a non-empty string")
                continue
            if not item.strip():
                errors.append(f"{source}: {field}[{index}] must be a non-empty string")
            if item in seen:
                errors.append(f"{source}: {field} contains duplicate value {item!r}")
            seen.add(item)


def _validate_known_semantics(payload: Dict[str, Any], source: str, errors: List[str]) -> None:
    """Validate cross-field contracts that a recursive type check cannot see.

    The JSON files intentionally remain extensible (upstream adapters can add
    implementation metadata), but the fields that define the statistical
    design are checked here so a typo cannot silently change the experiment.
    """

    for field in ("matrix_revision", "citation_cutoff"):
        value = payload.get(field)
        if isinstance(value, str) and not MATRIX_REVISION_RE.fullmatch(value):
            errors.append(f"{source}: {field} must use YYYY-MM-DD")
    policy = payload.get("concurrent_policy")
    if isinstance(policy, str) and policy != MATRIX_CONCURRENT_POLICY:
        errors.append(
            f"{source}: concurrent_policy must be {MATRIX_CONCURRENT_POLICY!r}"
        )

    arms = payload.get("arms")
    if isinstance(arms, list) and all(isinstance(item, str) for item in arms):
        if len(set(arms)) != len(arms):
            errors.append(f"{source}: arms must not contain duplicate values")

    # E1/E2 explicitly partition rows into main and supplementary tables.  A
    # missing row is much easier to catch at config validation than in a paper
    # table assembled months later.
    main_arms = payload.get("main_table_arms")
    supplementary_arms = payload.get("supplementary_arms")
    if main_arms is not None or supplementary_arms is not None:
        if not isinstance(main_arms, list) or not isinstance(supplementary_arms, list):
            errors.append(f"{source}: main_table_arms and supplementary_arms must be lists")
        elif isinstance(arms, list) and all(isinstance(item, str) for item in arms):
            main_set = set(main_arms)
            supp_set = set(supplementary_arms)
            if len(main_set) != len(main_arms) or len(supp_set) != len(supplementary_arms):
                errors.append(f"{source}: main/supplementary arm lists must be unique")
            if main_set & supp_set:
                errors.append(f"{source}: main and supplementary arm lists overlap")
            if main_set | supp_set != set(arms):
                errors.append(
                    f"{source}: main_table_arms + supplementary_arms must partition arms"
                )
            unknown = (main_set | supp_set) - set(arms)
            if unknown:
                errors.append(
                    f"{source}: table arm lists contain undeclared arms: {sorted(unknown)}"
                )

    # Final-matrix D0 is a 2x2 randomized design.  s2 is a readout stratum,
    # never a third randomized factor (the old eight-cell draft is rejected).
    if payload.get("id") == "D0":
        design = payload.get("design")
        if isinstance(design, dict):
            factors = design.get("factors")
            if isinstance(factors, dict):
                expected = {"signal_type", "action"}
                if set(factors) != expected:
                    errors.append(
                        f"{source}: D0 factors must be exactly signal_type and action"
                    )
                if factors.get("signal_type") != ["dl_top", "di_top"]:
                    errors.append(f"{source}: D0 signal_type levels are dl_top, di_top")
                if factors.get("action") != ["repair", "intervene"]:
                    errors.append(f"{source}: D0 action levels are repair, intervene")
            strata = design.get("s2_analysis_strata")
            if isinstance(strata, dict) and set(strata) >= {"low", "high", "role"}:
                role = strata.get("role", "")
                if "not_a_randomized_factor" not in str(role):
                    errors.append(
                        f"{source}: D0 s2_analysis_strata must be marked non-randomized"
                    )
            minimum = design.get("minimum_states")
            per_cell = design.get("minimum_states_per_cell")
            if isinstance(minimum, int) and isinstance(per_cell, int):
                if per_cell * 4 > minimum:
                    errors.append(
                        f"{source}: D0 four-cell minimum exceeds minimum_states"
                    )

    # E1's benchmark macro is exactly the seven-benchmark split from the final
    # matrix. Keep this check narrowly scoped so generic experiment configs can
    # still use their own evaluation schema.
    if payload.get("id") == "E1":
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, dict):
            b32 = evaluation.get("benchmarks_32_samples")
            b4 = evaluation.get("benchmarks_4_samples")
            mapping = evaluation.get("sampling_by_benchmark")
            count = evaluation.get("benchmark_count")
            if isinstance(b32, list) and isinstance(b4, list):
                expected_count = len(set(b32) | set(b4))
                if count != expected_count or count != 7:
                    errors.append(f"{source}: E1 benchmark_count must equal seven unique benchmarks")
                if isinstance(mapping, dict):
                    mapped = set(mapping.get("avg_at_32", [])) | set(mapping.get("avg_at_4", []))
                    if mapped != set(b32) | set(b4):
                        errors.append(
                            f"{source}: E1 sampling_by_benchmark must cover every benchmark exactly"
                        )
                    if set(mapping.get("avg_at_32", [])) != set(b32) or set(
                        mapping.get("avg_at_4", [])
                    ) != set(b4):
                        errors.append(
                            f"{source}: E1 sampling map does not match avg@32/avg@4 lists"
                        )
                    if any(
                        not isinstance(mapping.get(key), list)
                        or len(mapping.get(key, [])) != len(set(mapping.get(key, [])))
                        for key in ("avg_at_32", "avg_at_4")
                    ):
                        errors.append(f"{source}: E1 sampling lists must not contain duplicates")
            seed_policy = payload.get("seed_policy")
            if isinstance(seed_policy, dict) and isinstance(arms, list):
                declared = set(arms)
                for field in (
                    "core_trained_arms",
                    "mechanism_arms",
                    "evaluation_only_arms_without_training_seeds",
                ):
                    values = seed_policy.get(field, [])
                    if isinstance(values, list):
                        unknown = set(values) - declared
                        if unknown:
                            errors.append(
                                f"{source}: seed_policy.{field} has undeclared arms {sorted(unknown)}"
                            )
                partition = seed_policy.get("trained_arm_partition")
                if isinstance(partition, dict):
                    groups = []
                    for name, values in partition.items():
                        if not isinstance(values, list):
                            errors.append(f"{source}: trained_arm_partition.{name} must be a list")
                            continue
                        if len(values) != len(set(values)):
                            errors.append(f"{source}: trained_arm_partition.{name} contains duplicates")
                        groups.append((name, set(values)))
                    seen_partition: set[str] = set()
                    for name, values in groups:
                        overlap = seen_partition & values
                        if overlap:
                            errors.append(
                                f"{source}: trained_arm_partition overlaps at {sorted(overlap)}"
                            )
                        seen_partition |= values
                    if seen_partition != declared:
                        errors.append(
                            f"{source}: trained_arm_partition must cover every declared arm exactly once"
                        )
            success_gate = payload.get("success_gate")
            if isinstance(success_gate, dict):
                if success_gate.get("w2_point_estimate_margin_pp") != 1.5:
                    errors.append(f"{source}: E1 W2 point-estimate margin must be 1.5 pp")
                w2_ci_rule = success_gate.get("w2_ci_rule", "")
                if not isinstance(w2_ci_rule, str) or "lower_95ci" not in w2_ci_rule:
                    errors.append(f"{source}: E1 W2 must separately require a positive CI")

    if payload.get("id") in {"E1", "E2"}:
        routing_contract = payload.get("routing_contract")
        if not isinstance(routing_contract, dict):
            errors.append(f"{source}: routing_contract must be present")
        else:
            if routing_contract.get("primary_quantiles") != ROUTER_PRIMARY_QUANTILES:
                errors.append(
                    f"{source}: routing_contract.primary_quantiles must match D1 q80"
                )
            if routing_contract.get("threshold_scope") != "global_frozen":
                errors.append(
                    f"{source}: routing_contract.threshold_scope must be global_frozen"
                )

    if payload.get("id") == "D1":
        signals = payload.get("signals")
        if isinstance(signals, dict):
            scope = signals.get("router_threshold_scope")
            if scope != "global_frozen":
                errors.append(f"{source}: D1 router_threshold_scope must be global_frozen")
            primary_quantiles = signals.get("router_primary_quantiles")
            if primary_quantiles != ROUTER_PRIMARY_QUANTILES:
                errors.append(
                    f"{source}: D1 router_primary_quantiles must freeze s1/s2 at q80"
                )
            quantiles = signals.get("batch_robust_normalization_quantiles")
            if isinstance(quantiles, list) and quantiles != sorted(set(quantiles)):
                errors.append(f"{source}: D1 normalization quantiles must be sorted and unique")

    if payload.get("id") == "E2":
        seed_policy = payload.get("seed_policy")
        if isinstance(seed_policy, dict) and isinstance(arms, list):
            declared = set(arms)
            for field in (
                "trained_arms",
                "primary_main_trained_arms",
                "secondary_arms",
                "evaluation_only_arms_without_training_seeds",
            ):
                values = seed_policy.get(field, [])
                if isinstance(values, list) and (set(values) - declared):
                    errors.append(
                        f"{source}: seed_policy.{field} has undeclared arms "
                        f"{sorted(set(values) - declared)}"
                    )
            partition = seed_policy.get("trained_arm_partition")
            if isinstance(partition, dict):
                parts = []
                for name, values in partition.items():
                    if not isinstance(values, list):
                        errors.append(f"{source}: trained_arm_partition.{name} must be a list")
                        continue
                    if len(values) != len(set(values)):
                        errors.append(f"{source}: trained_arm_partition.{name} contains duplicates")
                    parts.append(set(values))
                union = set().union(*parts) if parts else set()
                if sum(len(part) for part in parts) != len(union) or union != declared:
                    errors.append(
                        f"{source}: trained_arm_partition must cover every declared arm exactly once"
                    )
        statistics = payload.get("statistics")
        if not isinstance(statistics, dict):
            errors.append(f"{source}: E2 statistics must be present")
        else:
            if statistics.get("rubric_interaction_excluded_labels") != ["MIXED"]:
                errors.append(f"{source}: E2 rubric interaction must exclude MIXED")
            if statistics.get("rubric_interaction_completion_source") != (
                "post_freeze_final_Full_pass_only"
            ):
                errors.append(
                    f"{source}: E2 rubric interaction must use post-freeze Full completions"
                )
            predictions = statistics.get("arm_level_predictions")
            required_predictions = {
                "repair_insertable",
                "intervene_insertable",
                "intervene_global_revision",
                "repair_global_revision",
            }
            if not isinstance(predictions, dict) or set(predictions) != required_predictions:
                errors.append(
                    f"{source}: E2 must declare all four arm-level rubric predictions"
                )


def _validate_frozen_experiment_contracts(
    payload: Dict[str, Any], source: str, errors: List[str]
) -> None:
    """Fail closed on the preregistered fields that determine experiment results."""

    experiment_id = payload.get("id")
    arms = payload.get("arms")

    if experiment_id == "A1-A8":
        _validate_exact_string_set(
            arms, f"{source}: A1-A8.arms", ABLATION_ARMS, errors
        )
        parameter_grids = payload.get("parameter_grids")
        if _validate_object_schema(
            parameter_grids,
            f"{source}: A1-A8.parameter_grids",
            {"A5", "A6"},
            {"A5", "A6"},
            errors,
        ):
            assert isinstance(parameter_grids, dict)
            _validate_fixed_values(
                parameter_grids.get("A5"),
                f"{source}: A1-A8.parameter_grids.A5",
                A5_FIXED,
                errors,
            )
        _validate_fixed_values(
            payload,
            f"{source}: A1-A8",
            {
                "protected_core": ABLATION_PROTECTED_CORE,
                "cut_first_order": ABLATION_CUT_FIRST_ORDER,
            },
            errors,
            allowed_keys=set(payload),
            required_keys={"protected_core", "cut_first_order"},
        )

    if experiment_id == "D0":
        _validate_exact_string_set(arms, f"{source}: D0.arms", D0_ARMS, errors)
        design = payload.get("design")
        if not _validate_object_schema(
            design,
            f"{source}: D0.design",
            D0_DESIGN_KEYS,
            D0_DESIGN_KEYS,
            errors,
        ):
            if not isinstance(design, dict):
                return
        assert isinstance(design, dict)
        _validate_fixed_values(
            design.get("factors"),
            f"{source}: D0.design.factors",
            {
                "signal_type": ["dl_top", "di_top"],
                "action": ["repair", "intervene"],
            },
            errors,
        )
        _validate_fixed_values(
            design.get("s2_analysis_strata"),
            f"{source}: D0.design.s2_analysis_strata",
            D0_S2_STRATA_FIXED,
            errors,
            allowed_keys=D0_S2_STRATA_KEYS,
            required_keys=D0_S2_STRATA_KEYS,
        )

    if experiment_id == "D1":
        _validate_fixed_values(
            payload.get("acceptance_gate_calibration"),
            f"{source}: D1.acceptance_gate_calibration",
            {
                "probe_rollouts_per_event": 4,
                "probe_aggregation": "paired_arithmetic_mean",
            },
            errors,
            allowed_keys=D1_ACCEPTANCE_GATE_KEYS,
            required_keys=D1_ACCEPTANCE_GATE_KEYS,
        )
        artifact_schema = payload.get("artifact_schema")
        _validate_fixed_values(
            artifact_schema,
            f"{source}: D1.artifact_schema",
            {
                "version": "rvi-d1-freeze-bundle-v3",
                "run_manifest_requires_both_artifact_hashes": True,
                "replay_must_be_read_only": True,
            },
            errors,
            allowed_keys=D1_ARTIFACT_SCHEMA_KEYS,
            required_keys=D1_ARTIFACT_SCHEMA_KEYS,
        )
        if isinstance(artifact_schema, dict):
            threshold_artifact = artifact_schema.get("threshold_artifact")
            _validate_fixed_values(
                threshold_artifact,
                f"{source}: D1.artifact_schema.threshold_artifact",
                {
                    "schema": "rvi-signals-v3",
                    "forbidden_field": "gate_artifact_sha256",
                },
                errors,
                allowed_keys=D1_THRESHOLD_ARTIFACT_KEYS,
                required_keys=D1_THRESHOLD_ARTIFACT_KEYS,
            )
            if isinstance(threshold_artifact, dict):
                threshold_hash_fields = threshold_artifact.get("required_hash_fields")
                if isinstance(threshold_hash_fields, list) and (
                    "gate_artifact_sha256" in threshold_hash_fields
                ):
                    errors.append(
                        f"{source}: D1 threshold artifact must not bind the gate hash"
                    )

            _validate_fixed_values(
                artifact_schema.get("joint_gate_artifact"),
                f"{source}: D1.artifact_schema.joint_gate_artifact",
                {
                    "schema": "rvi-joint-gate-v2",
                    "required_fields": [
                        "global_joint_gate_max_stat_q95",
                        "probe_rollouts_per_event",
                        "aggregation",
                        "calibration_split_sha256",
                        "threshold_artifact_sha256",
                        "code_revision",
                    ],
                    "probe_rollouts_per_event": 4,
                    "aggregation": "paired_arithmetic_mean",
                    "one_way_binding": (
                        "joint_gate_artifact_binds_threshold_artifact_sha256; "
                        "threshold_artifact_does_not_bind_gate_hash"
                    ),
                },
                errors,
                allowed_keys=D1_JOINT_GATE_ARTIFACT_KEYS,
                required_keys=D1_JOINT_GATE_ARTIFACT_KEYS,
            )

    if experiment_id == "E1":
        _validate_exact_string_set(arms, f"{source}: E1.arms", E1_ARMS, errors)
        _validate_exact_string_set(
            payload.get("main_table_arms"),
            f"{source}: E1.main_table_arms",
            E1_MAIN_ARMS,
            errors,
        )
        _validate_exact_string_set(
            payload.get("supplementary_arms"),
            f"{source}: E1.supplementary_arms",
            E1_SUPPLEMENTARY_ARMS,
            errors,
        )

        models = payload.get("models")
        _validate_fixed_values(
            models,
            f"{source}: E1.models",
            {
                "teacher": "Qwen/Qwen3-4B-Instruct-2507",
                "teacher_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                "student": "Qwen/Qwen3-1.7B",
                "student_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "student_lineage": "base_with_non_thinking_serializer",
                "shared_vocab_required": True,
            },
            errors,
            allowed_keys=E1_MODEL_KEYS,
            required_keys=E1_MODEL_KEYS,
        )
        if isinstance(models, dict):
            _validate_fixed_values(
                models.get("prompt_contract"),
                f"{source}: E1.models.prompt_contract",
                {
                    "canonical_serializer": "rvi_opd_non_thinking_v1",
                    "native_template_enable_thinking": False,
                    "canonical_serializer_thinking": False,
                    "stop_token_ids": [151645, 151643],
                    "pad_token_id": 151643,
                    "runtime_context_limit": 40960,
                    "validated_pretraining_context": 32768,
                },
                errors,
                allowed_keys=E1_PROMPT_CONTRACT_KEYS,
                required_keys=E1_PROMPT_CONTRACT_KEYS,
            )

        _validate_fixed_values(
            payload.get("training"),
            f"{source}: E1.training",
            E1_TRAINING_FIXED,
            errors,
        )
        evaluation_fixed = dict(E1_EVALUATION_FIXED)
        evaluation_fixed.update(
            {
                "benchmarks_32_samples": E1_BENCHMARKS_32,
                "benchmarks_4_samples": E1_BENCHMARKS_4,
                "benchmark_count": 7,
                "sampling_by_benchmark": {
                    "avg_at_32": E1_BENCHMARKS_32,
                    "avg_at_4": E1_BENCHMARKS_4,
                },
            }
        )
        _validate_fixed_values(
            payload.get("evaluation"),
            f"{source}: E1.evaluation",
            evaluation_fixed,
            errors,
            allowed_keys=E1_EVALUATION_KEYS,
            required_keys=E1_EVALUATION_KEYS,
        )
        seed_policy = payload.get("seed_policy")
        seed_policy_keys = set(E1_SEED_POLICY) | {"trained_arm_partition"}
        _validate_fixed_values(
            seed_policy,
            f"{source}: E1.seed_policy",
            E1_SEED_POLICY,
            errors,
            allowed_keys=seed_policy_keys,
            required_keys=seed_policy_keys,
        )
        if isinstance(seed_policy, dict):
            _validate_fixed_values(
                seed_policy.get("trained_arm_partition"),
                f"{source}: E1.seed_policy.trained_arm_partition",
                E1_TRAINED_ARM_PARTITION,
                errors,
            )

    if experiment_id == "E2":
        _validate_exact_string_set(arms, f"{source}: E2.arms", E2_ARMS, errors)
        _validate_exact_string_set(
            payload.get("main_table_arms"),
            f"{source}: E2.main_table_arms",
            E2_MAIN_ARMS,
            errors,
        )
        _validate_exact_string_set(
            payload.get("supplementary_arms"),
            f"{source}: E2.supplementary_arms",
            E2_SUPPLEMENTARY_ARMS,
            errors,
        )

        models = payload.get("models")
        _validate_fixed_values(
            models,
            f"{source}: E2.models",
            E2_MODEL_FIXED,
            errors,
            allowed_keys=E2_MODEL_KEYS,
            required_keys=E2_MODEL_KEYS,
        )
        if isinstance(models, dict):
            _validate_fixed_values(
                models.get("chat_template"),
                f"{source}: E2.models.chat_template",
                {
                    "native_template_enable_thinking": False,
                    "canonical_serializer_thinking": False,
                    "canonical_serializer": "rvi_opd_non_thinking_v1",
                    "stop_token_ids": [151645, 151643],
                    "pad_token_id": 151643,
                    "identical_for_teacher_student": False,
                    "identical_within_all_arms": True,
                    "runtime_context_limit_overrides_tokenizer_metadata": 40960,
                },
                errors,
                allowed_keys=E2_CHAT_TEMPLATE_KEYS,
                required_keys=E2_CHAT_TEMPLATE_KEYS,
            )

        _validate_fixed_values(
            payload.get("training"),
            f"{source}: E2.training",
            E2_TRAINING_FIXED,
            errors,
        )
        official = payload.get("official_evaluation")
        _validate_fixed_values(
            official,
            f"{source}: E2.official_evaluation",
            {
                "dataset": "openai/healthbench",
                "revision": "40ee1968852fc57f625934251ac22be47077a8fb",
                "datasets": ["healthbench_full", "healthbench_hard"],
                "full_prompt_count": 5000,
                "hard_prompt_count": 1000,
                "hard_is_subset_of_full": True,
                "reuse_full_completions_for_hard_subset": True,
                "do_not_modify_official_items_or_score": True,
            },
            errors,
            allowed_keys=E2_OFFICIAL_EVALUATION_KEYS,
            required_keys=E2_OFFICIAL_EVALUATION_KEYS,
        )
        if isinstance(official, dict):
            _validate_fixed_values(
                official.get("answer_generation"),
                f"{source}: E2.official_evaluation.answer_generation",
                E2_ANSWER_GENERATION_FIXED,
                errors,
                allowed_keys=E2_ANSWER_GENERATION_KEYS,
                required_keys=E2_ANSWER_GENERATION_KEYS,
            )
            _validate_fixed_values(
                official.get("staged_evaluation"),
                f"{source}: E2.official_evaluation.staged_evaluation",
                {
                    "pilot_prompt_count": 500,
                    "pilot_name": "resource_and_grader_behavior_pilot",
                    "pilot_purpose": (
                        "length_cost_latency_and_grading_consistency_before_final_matrix; "
                        "never_compare_candidate_arms_or_tune_router_loss_gate_or_claims"
                    ),
                    "pilot_is_not_confirmatory": True,
                    "pilot_checkpoint_scope": "one_locked_reference_or_Base_checkpoint_only",
                    "pilot_prerequisite": (
                        "rubric_annotation_manifest_and_adjudication_hash_frozen_"
                        "before_pilot_completion_generation"
                    ),
                    "pilot_outputs_excluded_from_final_arm_comparisons": True,
                    "final_full_evaluation": (
                        "one_pass_over_all_5000_full_prompts_after_all_settings_are_frozen"
                    ),
                    "hard_uses_indexed_full_completions": True,
                    "final_sampling_manifest_is_distinct_from_pilot": True,
                },
                errors,
                allowed_keys=E2_STAGED_EVALUATION_KEYS,
                required_keys=E2_STAGED_EVALUATION_KEYS,
            )

        seed_policy = payload.get("seed_policy")
        e2_seed_policy_keys = set(E2_SEED_POLICY) | {
            "trained_arm_partition",
            "aggregate_inference",
        }
        _validate_fixed_values(
            seed_policy,
            f"{source}: E2.seed_policy",
            E2_SEED_POLICY,
            errors,
            allowed_keys=e2_seed_policy_keys,
            required_keys=e2_seed_policy_keys,
        )
        if isinstance(seed_policy, dict):
            _validate_fixed_values(
                seed_policy.get("trained_arm_partition"),
                f"{source}: E2.seed_policy.trained_arm_partition",
                E2_TRAINED_ARM_PARTITION,
                errors,
            )

        rubric_annotation = payload.get("rubric_annotation")
        if isinstance(rubric_annotation, dict):
            _validate_fixed_values(
                rubric_annotation.get("sample"),
                f"{source}: E2.rubric_annotation.sample",
                {
                    "prompts": 500,
                    "sampling_frame": "healthbench_full",
                    "sample_seed": 20260828,
                    "label_all_rubric_items_for_selected_prompts": True,
                },
                errors,
                allowed_keys=E2_RUBRIC_SAMPLE_KEYS,
                required_keys=E2_RUBRIC_SAMPLE_KEYS,
            )

    if experiment_id in {"E1", "E2"}:
        routing = payload.get("routing_contract")
        _validate_fixed_values(
            routing,
            f"{source}: {experiment_id}.routing_contract",
            {
                "primary_quantiles": ROUTER_PRIMARY_QUANTILES,
                "analysis_band_quantiles": {"low": 0.25, "high": 0.75},
                "threshold_scope": "global_frozen",
                "trajectory_relative_ranks": "exploratory_only",
            },
            errors,
            allowed_keys=ROUTING_CONTRACT_KEYS,
            required_keys=ROUTING_CONTRACT_KEYS,
        )


def validate_experiment_config(
    payload: Any, source: str = "<memory>", *, run_ready: bool = False
) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return [f"{source}: top-level config must be an object"]

    missing = REQUIRED_EXPERIMENT_FIELDS - set(payload)
    if missing:
        errors.append(f"{source}: missing fields: {', '.join(sorted(missing))}")

    for field in REQUIRED_STRING_FIELDS:
        if field in payload:
            value = payload[field]
            if not isinstance(value, str) or (value and not value.strip()):
                errors.append(f"{source}: {field} must be a non-empty string")
    _validate_required_string_lists(payload, source, errors)
    _validate_id_required_fields(payload, source, errors)

    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            errors.append(f"{source}: top-level config contains an empty or non-string field name")
            continue
        _validate_value(
            value,
            f"{source}: {key}",
            errors,
            _is_probability_field(key),
            _is_integer_field(key),
            _is_numeric_field(key),
        )
    _validate_known_semantics(payload, source, errors)
    _validate_frozen_experiment_contracts(payload, source, errors)
    if run_ready:
        _validate_run_ready_values(payload, source, errors)
    return errors


def validate_config_paths(paths: Iterable[Path], *, run_ready: bool = False) -> List[str]:
    errors: List[str] = []
    seen_ids: Dict[str, Path] = {}
    matrix_revisions: Dict[str, Path] = {}
    citation_cutoffs: Dict[str, Path] = {}
    for path in paths:
        try:
            payload = load_config(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        errors.extend(validate_experiment_config(payload, str(path), run_ready=run_ready))
        experiment_id = payload.get("id")
        if isinstance(experiment_id, str) and experiment_id.strip():
            if experiment_id in seen_ids:
                errors.append(
                    f"{path}: duplicate experiment id {experiment_id!r}; "
                    f"first declared in {seen_ids[experiment_id]}"
                )
            else:
                seen_ids[experiment_id] = path
        for field, registry in (
            ("matrix_revision", matrix_revisions),
            ("citation_cutoff", citation_cutoffs),
        ):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                if registry and value not in registry:
                    errors.append(
                        f"{path}: {field} {value!r} disagrees with the value in "
                        f"{next(iter(registry.values()))}"
                    )
                else:
                    registry.setdefault(value, path)
    return errors


def validate_upstreams_lock(
    lock_path: Path, payloads: Iterable[Dict[str, Any]]
) -> List[str]:
    """Check model/data revisions in configs against the repository lock.

    This is a lightweight preflight (it does not download weights).  It catches
    the common failure mode where a readable model name is changed while the
    immutable revision in the lock remains untouched.
    """

    errors: List[str] = []
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{lock_path}: {exc}"]
    if not isinstance(lock, dict):
        return [f"{lock_path}: lock must be an object"]
    model_entries = lock.get("models", {})
    dataset_entries = lock.get("datasets", {})
    if not isinstance(model_entries, dict) or not isinstance(dataset_entries, dict):
        return [f"{lock_path}: models and datasets lock sections must be objects"]
    models_by_id = {
        entry.get("id"): entry
        for entry in model_entries.values()
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    datasets_by_id = {
        entry.get("id"): entry
        for entry in dataset_entries.values()
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    sha40 = re.compile(r"^[0-9a-f]{40}$")

    def check_model_pair(payload: Dict[str, Any], section: str) -> None:
        section_value = payload.get(section)
        if not isinstance(section_value, dict):
            return
        for key, value in section_value.items():
            if not key.endswith("_revision") or not isinstance(value, str):
                continue
            base_key = key[: -len("_revision")]
            model_id = section_value.get(base_key)
            if not isinstance(model_id, str):
                continue
            entry = models_by_id.get(model_id)
            if entry is None:
                errors.append(f"{section}.{base_key}: {model_id!r} is absent from {lock_path.name}")
            elif entry.get("revision") != value:
                errors.append(
                    f"{section}.{base_key}: revision {value!r} disagrees with lock "
                    f"{entry.get('revision')!r}"
                )
            if not sha40.fullmatch(value):
                errors.append(f"{section}.{key}: must be a 40-hex immutable revision")

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        check_model_pair(payload, "models")
        # Explicit dataset blocks have an ID and revision together.
        def check_dataset_block(block: Any, path: str = "data") -> None:
            if not isinstance(block, dict):
                return
            dataset_id = block.get("dataset") or block.get("train") or block.get("id")
            revision = block.get("revision")
            if isinstance(dataset_id, str) and isinstance(revision, str):
                entry = datasets_by_id.get(dataset_id)
                if entry is None:
                    errors.append(f"{path}: dataset {dataset_id!r} is absent from lock")
                elif entry.get("revision") != revision:
                    errors.append(
                        f"{path}: revision {revision!r} disagrees with lock "
                        f"{entry.get('revision')!r}"
                    )
            for key, child in block.items():
                if isinstance(child, dict):
                    check_dataset_block(child, f"{path}.{key}")
        check_dataset_block(payload.get("data"))
        check_dataset_block(payload.get("training_data"), "training_data")
        check_dataset_block(payload.get("official_evaluation"), "official_evaluation")
    return errors
