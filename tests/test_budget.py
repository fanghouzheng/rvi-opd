import unittest

from rvi_opd.budget import (
    BudgetLedger,
    LedgerEntry,
    assert_matched,
    audit_match,
    realized_repair_quota,
    select_frontloaded_triggers,
)
from rvi_opd.models import CostVector


class BudgetTests(unittest.TestCase):
    def test_ledger_never_refunds_rejected_intervention(self) -> None:
        ledger = BudgetLedger()
        ledger.add(
            LedgerEntry(
                "run",
                "rvi",
                "s1",
                "intervene",
                "repair",
                CostVector(
                    examples=1,
                    teacher_scored_tokens=12,
                    teacher_generated_tokens=8,
                    gate_teacher_scored_tokens=4,
                ),
            )
        )
        total = ledger.total("rvi")
        self.assertEqual(total.teacher_generated_tokens, 8)
        self.assertEqual(total.gate_teacher_scored_tokens, 4)

    def test_matcher_checks_declared_axes_only(self) -> None:
        left = CostVector(examples=2, teacher_scored_tokens=10)
        right = CostVector(examples=2, teacher_scored_tokens=10, teacher_generated_tokens=10)
        assert_matched(left, right, ["examples", "teacher_scored_tokens"])
        mismatches = audit_match(left, right, ["teacher_generated_tokens"])
        self.assertEqual(len(mismatches), 1)

    def test_matcher_raises(self) -> None:
        with self.assertRaises(AssertionError):
            assert_matched(
                CostVector(teacher_scored_tokens=9),
                CostVector(teacher_scored_tokens=10),
                ["teacher_scored_tokens"],
            )

    def test_frontloading_and_valid_trigger_cooldown(self) -> None:
        self.assertEqual(select_frontloaded_triggers([2, 5, 7, 11, 13], 2, 1), [2, 7])

    def test_realized_bridge_length_drives_repair_quota(self) -> None:
        self.assertEqual(realized_repair_quota([20, 31, 14]), 65)
        for invalid in ([1.5], [-1], [True]):
            with self.assertRaises(ValueError):
                realized_repair_quota(invalid)

    def test_matcher_rejects_invalid_tolerance(self) -> None:
        for tolerance in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                audit_match(
                    CostVector(examples=1),
                    CostVector(examples=1),
                    ["examples"],
                    {"examples": tolerance},
                )

    def test_cost_vector_rejects_negative_nan_and_fractional_counts(self) -> None:
        for kwargs in (
            {"teacher_scored_tokens": -1},
            {"teacher_scored_tokens": 1.5},
            {"teacher_gpu_seconds": float("nan")},
            {"wall_time_ms": float("inf")},
        ):
            with self.assertRaises(ValueError):
                CostVector(**kwargs)

    def test_grouped_totals_do_not_cancel_across_strata(self) -> None:
        ledger = BudgetLedger(
            [
                LedgerEntry(
                    "r", "repair", "s1", "repair", "repair", CostVector(examples=1), "a", 1
                ),
                LedgerEntry(
                    "r", "repair", "s2", "repair", "repair", CostVector(examples=2), "b", 1
                ),
            ]
        )
        totals = ledger.grouped_totals("repair")
        self.assertEqual(totals[("r", "a", 1)].examples, 1)
        self.assertEqual(totals[("r", "b", 1)].examples, 2)


if __name__ == "__main__":
    unittest.main()
