from __future__ import annotations

import random
from math import comb
from statistics import mean
from typing import List, Mapping, Sequence

from .models import BootstrapResult
from .signals import quantile


def _cluster_means(values: Mapping[str, Sequence[float]]) -> Mapping[str, float]:
    output = {}
    for cluster, observations in values.items():
        if not observations:
            raise ValueError(f"cluster {cluster!r} is empty")
        output[cluster] = mean(observations)
    return output


def paired_cluster_bootstrap(
    baseline: Mapping[str, Sequence[float]],
    treatment: Mapping[str, Sequence[float]],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Problem-clustered paired percentile bootstrap of treatment minus baseline."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if set(baseline) != set(treatment):
        raise ValueError("baseline and treatment must contain identical paired clusters")
    if not baseline:
        raise ValueError("at least one cluster is required")

    baseline_means = _cluster_means(baseline)
    treatment_means = _cluster_means(treatment)
    clusters = sorted(baseline)
    unequal = [key for key in clusters if len(baseline[key]) != len(treatment[key])]
    if unequal:
        raise ValueError(
            "paired clusters must contain equal observation counts: " + ", ".join(unequal)
        )
    differences = [treatment_means[key] - baseline_means[key] for key in clusters]
    estimate = mean(differences)
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(resamples):
        sampled = [differences[rng.randrange(len(differences))] for _ in clusters]
        draws.append(mean(sampled))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=estimate,
        lower=quantile(draws, alpha),
        upper=quantile(draws, 1.0 - alpha),
        confidence=confidence,
        clusters=len(clusters),
        resamples=resamples,
        seed=seed,
    )


def difference_in_differences(
    dl_repair: float, dl_intervene: float, di_repair: float, di_intervene: float
) -> float:
    """Interaction: action advantage in D^I minus action advantage in D^L."""

    return (di_intervene - di_repair) - (dl_intervene - dl_repair)


def three_way_action_interaction(
    dl_low_repair: float,
    dl_low_intervene: float,
    dl_high_repair: float,
    dl_high_intervene: float,
    di_low_repair: float,
    di_low_intervene: float,
    di_high_repair: float,
    di_high_intervene: float,
) -> float:
    """Return the action × signal-type × s2-band interaction.

    Positive values mean that the intervene-minus-repair advantage grows more
    from low to high s2 in D^I states than it does in D^L states.  This explicit
    cell-mean definition avoids relying on software-specific contrast coding.
    """

    dl_change = (dl_high_intervene - dl_high_repair) - (
        dl_low_intervene - dl_low_repair
    )
    di_change = (di_high_intervene - di_high_repair) - (
        di_low_intervene - di_low_repair
    )
    return di_change - dl_change


def ci_is_equivalent(lower: float, upper: float, margin: float) -> bool:
    """Check a preregistered TOST decision from its two-sided 90% CI.

    For alpha=0.05 TOST, equivalence holds exactly when the 90% confidence
    interval lies strictly inside ``[-margin, margin]``.
    """

    if margin <= 0:
        raise ValueError("equivalence margin must be positive")
    if lower > upper:
        raise ValueError("lower confidence bound cannot exceed upper bound")
    return lower > -margin and upper < margin


def mcnemar_exact_p(baseline: Sequence[int], treatment: Sequence[int]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired binary outcomes must be non-empty and equal length")
    if any(value not in (0, 1, False, True) for value in [*baseline, *treatment]):
        raise ValueError("McNemar outcomes must be binary")
    baseline_only = sum(left == 1 and right == 0 for left, right in zip(baseline, treatment))
    treatment_only = sum(left == 0 and right == 1 for left, right in zip(baseline, treatment))
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(baseline_only, treatment_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_seed_problem_bootstrap(
    baseline: Mapping[str, Mapping[str, Sequence[float]]],
    treatment: Mapping[str, Mapping[str, Sequence[float]]],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap over train seeds, problems, then paired rollouts.

    Train seeds are sampled first.  Within every sampled seed, problems are
    sampled; rollout indices stay paired within a sampled problem.  This is the
    E1 hierarchy and prevents treating rollout samples as independent problems.
    """

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not baseline or set(baseline) != set(treatment):
        raise ValueError("baseline and treatment must contain identical train seeds")

    seeds = sorted(baseline)
    seed_problem_differences = {}
    for train_seed in seeds:
        if not baseline[train_seed] or set(baseline[train_seed]) != set(treatment[train_seed]):
            raise ValueError(f"seed {train_seed!r} must contain identical non-empty problems")
        seed_problem_differences[train_seed] = {}
        for problem in sorted(baseline[train_seed]):
            left = baseline[train_seed][problem]
            right = treatment[train_seed][problem]
            if not left or len(left) != len(right):
                raise ValueError(
                    f"seed/problem {train_seed!r}/{problem!r} needs equal paired rollouts"
                )
            seed_problem_differences[train_seed][problem] = [
                float(right_value) - float(left_value)
                for left_value, right_value in zip(left, right)
            ]

    estimate = mean(
        mean(mean(rollouts) for rollouts in seed_problem_differences[train_seed].values())
        for train_seed in seeds
    )
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(resamples):
        sampled_seed_means: List[float] = []
        for _ in seeds:
            sampled_seed = seeds[rng.randrange(len(seeds))]
            problems = sorted(seed_problem_differences[sampled_seed])
            sampled_problem_means: List[float] = []
            for _ in problems:
                sampled_problem = problems[rng.randrange(len(problems))]
                rollout_differences = seed_problem_differences[sampled_seed][sampled_problem]
                sampled_rollouts = [
                    rollout_differences[rng.randrange(len(rollout_differences))]
                    for _ in rollout_differences
                ]
                sampled_problem_means.append(mean(sampled_rollouts))
            sampled_seed_means.append(mean(sampled_problem_means))
        draws.append(mean(sampled_seed_means))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=estimate,
        lower=quantile(draws, alpha),
        upper=quantile(draws, 1.0 - alpha),
        confidence=confidence,
        clusters=len(seeds),
        resamples=resamples,
        seed=seed,
    )


def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> Mapping[str, bool]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rejected = {name: False for name in p_values}
    still_rejecting = True
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        threshold = alpha / (total - index)
        if still_rejecting and value <= threshold:
            rejected[name] = True
        else:
            still_rejecting = False
    return rejected
