from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import BudgetMismatch, CostVector


@dataclass(frozen=True)
class LedgerEntry:
    run_id: str
    arm: str
    state_id: str
    requested_action: str
    effective_action: str
    cost: CostVector
    stratum: str = "default"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("run_id", "arm", "state_id", "requested_action", "effective_action", "stratum"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


class BudgetLedger:
    def __init__(self, entries: Iterable[LedgerEntry] = ()) -> None:
        self._entries: List[LedgerEntry] = list(entries)

    def add(self, entry: LedgerEntry) -> None:
        if not isinstance(entry, LedgerEntry):
            raise TypeError("entry must be a LedgerEntry")
        self._entries.append(entry)

    @property
    def entries(self) -> Tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    def total(self, arm: str) -> CostVector:
        total = CostVector()
        for entry in self._entries:
            if entry.arm == arm:
                total = total + entry.cost
        return total

    def grouped_totals(self, arm: str) -> Mapping[Tuple[str, str, int], CostVector]:
        totals: Dict[Tuple[str, str, int], CostVector] = {}
        for entry in self._entries:
            if entry.arm != arm:
                continue
            key = (entry.run_id, entry.stratum, entry.seed)
            totals[key] = totals.get(key, CostVector()) + entry.cost
        return totals

    def arms(self) -> Tuple[str, ...]:
        return tuple(sorted({entry.arm for entry in self._entries}))


def audit_match(
    left: CostVector,
    right: CostVector,
    match_on: Sequence[str],
    tolerances: Optional[Mapping[str, float]] = None,
) -> List[BudgetMismatch]:
    tolerances = tolerances or {}
    available = set(left.to_dict())
    unknown = set(match_on) - available
    if unknown:
        raise ValueError("unknown cost fields: " + ", ".join(sorted(unknown)))
    mismatches: List[BudgetMismatch] = []
    for field in match_on:
        left_value = float(getattr(left, field))
        right_value = float(getattr(right, field))
        tolerance = float(tolerances.get(field, 0.0))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"tolerance for {field} must be finite and non-negative")
        difference = abs(left_value - right_value)
        if difference > tolerance:
            mismatches.append(
                BudgetMismatch(
                    field=field,
                    left=left_value,
                    right=right_value,
                    absolute_difference=difference,
                    tolerance=tolerance,
                )
            )
    return mismatches


def assert_matched(
    left: CostVector,
    right: CostVector,
    match_on: Sequence[str],
    tolerances: Optional[Mapping[str, float]] = None,
) -> None:
    mismatches = audit_match(left, right, match_on, tolerances)
    if mismatches:
        details = "; ".join(
            f"{item.field}: {item.left} != {item.right} (tol={item.tolerance})"
            for item in mismatches
        )
        raise AssertionError("budget mismatch: " + details)


def select_frontloaded_triggers(
    trigger_indices: Sequence[int], max_takeovers: int, cooldown_valid_triggers: int
) -> List[int]:
    """Select the earliest triggers and skip a fixed count of subsequent valid triggers."""

    if max_takeovers < 0 or cooldown_valid_triggers < 0:
        raise ValueError("max_takeovers and cooldown must be non-negative")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in trigger_indices
    ):
        raise ValueError("trigger indices must be non-negative integers")
    ordered = sorted(set(trigger_indices))
    selected: List[int] = []
    cursor = 0
    while cursor < len(ordered) and len(selected) < max_takeovers:
        selected.append(ordered[cursor])
        cursor += cooldown_valid_triggers + 1
    return selected


def realized_repair_quota(bridge_token_lengths: Sequence[int]) -> int:
    """Number of repair-scored positions needed to match realized bridge queries."""

    if any(
        isinstance(length, bool) or not isinstance(length, int) or length < 0
        for length in bridge_token_lengths
    ):
        raise ValueError("bridge lengths must be non-negative integers")
    return sum(bridge_token_lengths)
