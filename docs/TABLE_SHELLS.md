# Preregistered table shells

## D0 factorial mechanism table (2×2)

The table crosses two prespecified signal strata with two actions. Signal type
is an eligibility/stratification descriptor; action is randomized within blocked
strata. The s2 high/low bands are prespecified subgroups, not additional
randomized cells. Ties and the frozen middle band are excluded before action
assignment.

| Signal | Action | states/prompts | local KL Δ | s2 residual AUC | verifier pass | teacher scored tok | teacher GPU-s |
|---|---|---:|---:|---:|---:|---:|---:|
| D^L top | repair | | | | | | |
| D^L top | intervene | | | | | | |
| D^I top | repair | | | | | | |
| D^I top | intervene | | | | | | |

For every row, report the prespecified s2-high and s2-low subgroup counts and
effects (including the `D^I` high-s2 surface-repair check). Report the primary
`Delta2 = (μ_I,DI − μ_R,DI) − (μ_I,DL − μ_R,DL)`, its problem-clustered 95% CI
and randomization p-value, plus the subgroup contrasts and their declared
multiplicity adjustment. Report forced-action ITT compliance and budget error on
target/scored positions, supervised tokens, and optimizer steps. Prefill/decode/
gate tokens, forward calls, and GPU-seconds are separate measurements and must
not be labelled matched.

## D2 paired continuation

| Arm | context changed? | local KL Δ | s2 Δ | TOST result | verifier Δ | time-to-recovery |
|---|---|---:|---:|---|---:|---:|
| base | no | | | | | |
| repair micro-update | no | | | | | |
| bridge | yes | | | | | |

## D3 detached

| Arm | bridge hash | post-leg context hash relation | s2 residual | verifier | final task | cost match |
|---|---|---|---:|---:|---:|---|
| normal | | differs from original | | | | |
| detached | same as normal | equals original | | | | |
| repair | n/a | equals original | | | | |

## E1 main table (seven benchmarks)

Teacher: `Qwen/Qwen3-4B-Instruct-2507`; student: `Qwen/Qwen3-1.7B` with the
locked non-thinking serializer. AIME24/25/26, AMC23, and HMMT-Feb26 are avg@32;
MATH500 and OlympiadBench are avg@4. These are arithmetic means of independent
correctness indicators, not pass@k.

| Method | AIME24 | AIME25 | AIME26 | MATH500 | AMC23 | OlympiadBench | HMMT-Feb26 | Macro | MMLU Δ | teacher GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | | | | |
| Teacher upper bound | | | | | | | | | | |
| SFT | | | | | | | | | | |
| Vanilla OPD | | | | | | | | | | |
| FastOPD (Relay fixed-prefix) | | | | | | | | | | |
| SKD | | | | | | | | | | |
| TA-OPD | | | | | | | | | | |
| TIP-select | | | | | | | | | | |
| Relay-OPD | | | | | | | | | | |
| TRD (full-vocabulary reference) | | | | | | | | | | |
| RvI-OPD | | | | | | | | | | |

KD/top-128, detached, repair-only, intervene-only, A2 action-shuffled, and GRPO
are supplementary rows. Every value identifies training-seed and sampling counts;
Base/Teacher are evaluation-only and have no training seeds.

## E2 main and rubric table

Teacher: `Qwen/Qwen3-4B-Instruct-2507`; student: `Qwen/Qwen3-0.6B`. HealthBench
Full is the primary official evaluation (5,000 prompts); Hard is an overlapping
1,000-prompt subset whose completions are indexed from Full and is never pooled
as independent data.

| Method | Full official | Hard official | INSERTABLE | GLOBAL_REVISION | negative violation | contradiction | teacher GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | |
| Teacher upper bound | | | | | | | |
| SFT | | | | | | | |
| Vanilla OPD | | | | | | | |
| FastOPD (Relay fixed-prefix) | | | | | | | |
| SKD | | | | | | | |
| TA-OPD | | | | | | | |
| TIP-select | | | | | | | |
| Relay-OPD | | | | | | | |
| TRD reference (full-vocabulary FKL) | | | | | | | |
| RvI-OPD | | | | | | | |

Repair-only, intervene-only, and A2 action-shuffled are supplementary rows.
Official scores and auxiliary rubric statistics are never merged into one
number. Report the preregistered difference-in-differences
`[(RvI−repair)_GLOBAL_REVISION] − [(RvI−repair)_INSERTABLE]` with its
Holm-adjusted prompt-clustered CI. The rubric columns use only the frozen
annotation manifest's post-freeze Full completions and exclude MIXED. Also
report repair/Base and intervene/Base INSERTABLE contrasts, intervene/Base
GLOBAL_REVISION, and the 90% TOST for repair/Base GLOBAL_REVISION with the
absolute 0.01 equivalence margin. Negative-item violation is a separate metric;
contradiction and generic-safety-boilerplate detection remain auxiliary until
their judge/template or descriptive coding contract is frozen.

## A1–A8 ablation shell

| ID | Prespecified contrast |
|---|---|
| A1 | force all states to repair (`d=0`) |
| A2 | constrained within-block action-payload shuffle (including realized length and cost) |
| A3 | D3 detached bridge (teacher leg supervised but not inserted into context) |
| A4 | signal ablation: s1-only, s2-only, and TIP-style `(1−ĥ)δ` control |
| A5 | threshold; frozen teacher-GPU-second budget multipliers `{0.5,1,2}`; and `L∈{1,3,5}` sweeps |
| A6 | repair loss: top-K FCE, full-vocabulary FKL, and RKL reweighting (same-support equivalent gradients are not duplicate arms) |
| A7 | degradation diagnostics: repetition, length inflation, missing `\\boxed{}`, entropy, and avg@k/pass@k |
| A8 | efficiency: teacher forward calls, wall-clock, memory, and GPU-seconds for Relay versus TRD |
