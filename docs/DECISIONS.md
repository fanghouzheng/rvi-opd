# Design decisions and resolved ambiguities

## HealthBench-first is a prospective execution amendment

The `healthbench-first` branch changes resource order, not the scientific matrix: the amendment was recorded with `scientific_outcomes_seen_before_amendment=false`, keeps the 2026-08-01 citation cutoff, and freezes the E2 config plus the complete math-config bundle before final HealthBench outputs are visible. Its core evidence is seven trained E2 arms at three seeds (21 runs), followed by one Full evaluation pass with Hard indexed from the same completions.

Math is released only if every condition in the immutable `healthbench-first-v1` intersection-union gate passes. As an independent necessary condition, RvI's paired HealthBench Full official-score delta against frozen Base must have estimate at least `+0.01` and a seed→prompt bootstrap lower 95% confidence bound above zero; beating every trained baseline cannot compensate for failing this check. The gate also includes RvI comparisons against the three non-oracle baselines, both single-action baselines and A2, the rubric mechanism checks, leave-one-seed-out stability and the negative-violation safety veto. A pass produces `GO_MATH`; any failure produces `STOP_AFTER_HEALTHBENCH` and all math targets are recorded as `NOT_RUN_HEALTHBENCH_GATE`. Optional E2 rows, a separate Hard sample or extra seeds cannot rescue a STOP.

This design deliberately treats HealthBench as a binary resource signal, not a tuning set. Math thresholds, hyperparameters, data splits, seeds, baselines, budgets and ablations cannot depend on its observed outcomes, and math still runs its own domain-specific D1/D2 after release. The full operational contract is in [`HEALTHBENCH_FIRST_PLAN.zh-CN.md`](HEALTHBENCH_FIRST_PLAN.zh-CN.md).

## D0 is a 2×2 mechanism test with an s2 subgroup readout

The final matrix reports the two prespecified signal strata (`D^L-top` and `D^I-top`) crossed with repair/intervene. Only action is randomized within the blocked signal strata; signal type is an eligibility/stratification descriptor, not an assigned treatment. The high/low s2 bands are frozen, prespecified analysis subgroups (with a middle band and ties excluded before action assignment); s2 is not a third randomized factor. The confirmatory estimand is therefore action-effect heterogeneity, `Delta2`, across signal strata. We report the high-s2 and low-s2 action contrasts, with the subgroup family and its multiplicity rule made explicit, but do not promote a three-way `Delta3` claim from this design.

The causal mechanism estimand uses isolated frozen-prefix copies with forced randomized actions, no gate and no shared cross-state update. Full policy training is a separate independent-run check. Mixing online parameter updates into state-level randomization would create interference and pseudoreplication.

## Final matrix scope

E1's confirmatory evaluation has seven benchmarks: AIME2024, AIME2025, AIME2026, AMC2023, HMMT-Feb2026, MATH500, and OlympiadBench (the first five use avg@32 and the latter two avg@4). E2 fixes the teacher/student pair to `Qwen/Qwen3-4B-Instruct-2507 → Qwen/Qwen3-0.6B`; HealthBench Full and Hard are evaluation-only, with Hard an overlapping subset of Full. The preregistered ablation set is A1–A8; no A9/A10 rows are part of the final matrix.

## D1 routing anchors are global and frozen

TA-style batch q05/q95 normalization remains a state descriptor/reproduction diagnostic. Router decisions instead use one raw-D/raw-C q05/q95 transform frozen globally on D1; inference batch composition and realized/future response length never change a threshold. Known prefix position is a stability diagnostic/covariate only.

The primary router thresholds `tau1/tau2` are the D1 global s1/s2 q80 values. The q25/q75 values are separate frozen boundaries for low/high subgroup reporting and D0 signal-cell eligibility; they are not alternative router thresholds. A5 may sweep q70/q75/q80, but only after q80 has been frozen as the primary setting.

## D2 repair is a temporary micro-update

A loss target alone cannot change continuation from an unchanged frozen model. For the mechanism probe, repair means a fixed-context FCE micro-update on a temporary adapter copy, followed by freezing and continuation from the original prefix. The trained repair-only checkpoint supplies a separate external-validity check. Forced teacher tokens are not called repair because they change context.

## FCE and same-support FKL are not separate arms

Teacher-top-K cross-entropy and teacher-top-K forward KL differ by teacher entropy only when teacher `q` is renormalized on the same support while student log probabilities come from its full softmax. Under that contract the entropy is constant with respect to student parameters. A6 therefore compares support definitions and KL directions rather than presenting equivalent gradients as independent methods.

## “Strict budget matching” has named axes

No scalar “teacher-token ratio” can simultaneously express teacher scoring, autoregressive generation, prefill, gate and inserted context. D0/D3 match target/scored-position counts, supervised tokens and optimizer steps within 1%; `teacher_scored_tokens` is a query-position equivalent, not a FLOP equivalence. Generated/inserted tokens, forward calls and GPU-seconds are reported separately. Only a separate discarded-sham-generation control can match generation/prefill, and even it must use measured GPU-seconds before making a compute claim.

## Variable paragraphs are not budgets

`L=3` is an algorithmic structural setting, not a compute unit. Matching and Pareto curves use realized tokens and GPU-seconds. Every leg also has a hard token cap.

## Gate costs never disappear

Rejected intervention rolls back behavior to repair but does not refund teacher generation or gate scoring. The ledger records requested and effective action separately.

## TRD diagnostic magnitudes are not thresholds

The cited epistemic-mass magnitudes were measured in a particular OPSD/reference-conditioned setup. RvI uses the phenomenon as a signal definition, then calibrates domain/model-specific thresholds on D1.

## TRD and Relay token lexicons remain separate

TRD's 16 epistemic-onset phrases define s2; Relay's 13 reflection bases define the top-1/top-K handoff trigger. They overlap but are not interchangeable. Each tokenizer produces two independently hashed ID artifacts.

Relay `phi(c)` requires both the teacher global argmax to be a single-token Relay ID and the student's top-5 to contain no Relay ID. The canonical baseline also keeps M=2, L=3, cap=256, no cooldown, M-th-leg termination and actual emitted-token k1 RKL. D3's M=1 forced-resume contract is a causal probe, not a claim about canonical Relay execution.

## HealthBench negative items use a separate violation metric

Official HealthBench score is preserved for the complete evaluation. A negative-only subset can have no positive denominator under the official normalization, so negative criteria are analyzed via weighted violation/satisfaction rates instead of a fabricated official score.

## Upstream code is not vendored

Where an upstream repository has no explicit compatible license, this repository cites the paper and reimplements formulas from the publication. Model/data/code revisions are recorded, but external source files are not copied.
