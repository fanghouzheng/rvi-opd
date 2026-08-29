from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

from .budget import BudgetLedger, LedgerEntry, assert_matched, select_frontloaded_triggers
from .gate import GateConfig, evaluate_intervention_gate
from .io import atomic_write_json
from .models import CostVector, RawStateSignal
from .router import action_counts, calibrate_thresholds, route_batch
from .signals import apply_frozen_scale, fit_frozen_scale
from .stats import difference_in_differences, paired_cluster_bootstrap


def _synthetic_states() -> List[RawStateSignal]:
    rows = [
        # Locally absorbable: high disagreement, high compatibility, low onset mass.
        ("absorb-1", "p1", "t1", 4, 1.20, 0.96, 0.0010, 0.01, 0.8),
        ("absorb-2", "p2", "t2", 8, 1.05, 0.92, 0.0012, 0.02, 0.7),
        # Prefix-failed: high disagreement, low compatibility, high onset mass.
        ("failed-1", "p3", "t3", 5, 1.30, 0.08, 0.0120, 0.05, 0.2),
        ("failed-2", "p4", "t4", 7, 1.10, 0.12, 0.0100, 0.04, 0.2),
        # Discardable.
        ("quiet-1", "p5", "t5", 3, 0.10, 0.50, 0.0008, 0.03, 0.8),
        ("quiet-2", "p6", "t6", 6, 0.12, 0.45, 0.0007, 0.02, 0.9),
        # Degenerate-loop and gate-failure fixtures are diagnostic, not main-router overrides.
        ("loop-1", "p7", "t7", 9, 0.08, 0.48, 0.0006, 0.92, 0.0),
        ("gate-fail-1", "p8", "t8", 2, 1.25, 0.10, 0.0110, 0.04, 0.1),
    ]
    states = [RawStateSignal(*row) for row in rows]
    intervention_eligible = {"failed-1", "failed-2", "gate-fail-1"}
    return [
        replace(
            state,
            relay_phi_eligible=True,
            intervention_budget_available=True,
            intervention_cooldown_available=True,
        )
        if state.state_id in intervention_eligible
        else state
        for state in states
    ]


def run_smoke(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_states = _synthetic_states()
    scale = fit_frozen_scale(raw_states, q_low=0.0, q_high=1.0)
    states = apply_frozen_scale(raw_states, scale)
    thresholds = calibrate_thresholds(
        states, s1_quantile=0.60, s2_quantile=0.60, frozen_scale=scale
    )
    decisions = route_batch(states, thresholds)

    accepted_gate = evaluate_intervention_gate(
        s2_before=[0.012, 0.011, 0.013, 0.012],
        s2_after=[0.003, 0.002, 0.004, 0.003],
        teacher_preferred_before=[0.10, 0.12, 0.11, 0.09],
        teacher_preferred_after=[0.55, 0.58, 0.52, 0.56],
    )
    rejected_gate = evaluate_intervention_gate(
        s2_before=[0.011, 0.010, 0.012, 0.011],
        s2_after=[0.011, 0.011, 0.012, 0.011],
        teacher_preferred_before=[0.20, 0.21, 0.19, 0.20],
        teacher_preferred_after=[0.21, 0.20, 0.20, 0.20],
        config=GateConfig(min_rollouts=4, min_s2_drop=0.001, min_teacher_preferred_gain=0.05),
    )

    ledger = BudgetLedger()
    matched_cost = CostVector(
        examples=4,
        teacher_scored_tokens=64,
        student_supervised_tokens=64,
        teacher_forward_calls=64,
        optimizer_steps=4,
    )
    ledger.add(
        LedgerEntry("smoke", "repair", "aggregate", "repair", "repair", matched_cost)
    )
    ledger.add(
        LedgerEntry(
            "smoke",
            "intervene",
            "aggregate",
            "intervene",
            "intervene",
            CostVector(
                examples=4,
                teacher_scored_tokens=64,
                teacher_generated_tokens=64,
                teacher_inserted_tokens=64,
                student_supervised_tokens=64,
                teacher_forward_calls=64,
                optimizer_steps=4,
                gate_teacher_scored_tokens=32,
            ),
        )
    )
    matched_fields = [
        "examples",
        "teacher_scored_tokens",
        "student_supervised_tokens",
        "optimizer_steps",
    ]
    assert_matched(ledger.total("repair"), ledger.total("intervene"), matched_fields)

    bootstrap = paired_cluster_bootstrap(
        baseline={"p1": [0.0, 0.0], "p2": [0.0, 1.0], "p3": [0.0, 0.0]},
        treatment={"p1": [1.0, 1.0], "p2": [1.0, 1.0], "p3": [0.0, 1.0]},
        resamples=2_000,
        seed=17,
    )
    report: Dict[str, object] = {
        "warning": "Synthetic smoke results validate software contracts only; they are not scientific evidence.",
        "frozen_scale_sha256": scale.artifact_sha256,
        "thresholds": thresholds.to_dict(),
        "route_counts": dict(action_counts(decisions)),
        "routes": [
            {**asdict(decision), "action": decision.action.value} for decision in decisions
        ],
        "frontloaded_triggers": select_frontloaded_triggers([2, 5, 7, 11, 13], 2, 1),
        "gate": {
            "accepted": {
                **asdict(accepted_gate),
                "requested_action": accepted_gate.requested_action.value,
                "effective_action": accepted_gate.effective_action.value,
            },
            "rolled_back": {
                **asdict(rejected_gate),
                "requested_action": rejected_gate.requested_action.value,
                "effective_action": rejected_gate.effective_action.value,
            },
        },
        "budget": {
            "match_on": matched_fields,
            "repair": ledger.total("repair").to_dict(),
            "intervene": ledger.total("intervene").to_dict(),
            "matched": True,
            "note": "Inserted/generated teacher tokens are mechanism descriptors, not matchable repair costs.",
        },
        "paired_bootstrap": asdict(bootstrap),
        "d0_difference_in_differences_fixture": difference_in_differences(
            dl_repair=0.62,
            dl_intervene=0.54,
            di_repair=0.20,
            di_intervene=0.74,
        ),
    }
    atomic_write_json(output_dir / "report.json", report)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "run_id": "smoke",
            "mode": "deterministic_synthetic",
            "seed": 17,
            "artifacts": ["report.json"],
            "success": True,
        },
    )
    return report
