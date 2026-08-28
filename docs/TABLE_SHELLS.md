# Preregistered table shells

## D0 factorial mechanism table

| Signal | s2 band | Action | states/prompts | local KL Δ | s2 residual AUC | verifier pass | teacher scored tok | teacher GPU-s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D^L top | low | repair | | | | | | |
| D^L top | low | intervene | | | | | | |
| D^L top | high | repair | | | | | | |
| D^L top | high | intervene | | | | | | |
| D^I top | low | repair | | | | | | |
| D^I top | low | intervene | | | | | | |
| D^I top | high | repair | | | | | | |
| D^I top | high | intervene | | | | | | |

Below the table report the exact `Delta3`, its problem-clustered 95% CI/randomization p-value, forced-action ITT compliance, and budget error on target/scored positions, supervised tokens and optimizer steps. Report prefill/decode/gate tokens, forward calls and GPU-seconds separately; do not label them matched.

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

## E1 main table

| Method | AIME24 | AIME25 | AIME26 | MATH500 | AMC23 | Olympiad | HMMT-Feb26 | HMMT-Nov25 | Macro | MMLU Δ | teacher GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | | | | | |
| Vanilla OPD | | | | | | | | | | | |
| TA-OPD | | | | | | | | | | | |
| Relay-OPD | | | | | | | | | | | |
| RvI-OPD | | | | | | | | | | | |
| A2 shuffled | | | | | | | | | | | |

Every value must identify train seed count and sampling count. Columns are `avg@32` or `avg@4` (not pass@k). Canonical TRD and Relay top-128 TRD use distinct row names; Base/Teacher are evaluation-only and have no training seeds.

## E2 main and rubric table

| Method | Full official | Hard official | INSERTABLE | GLOBAL_REVISION | negative violation | contradiction | teacher GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | |
| Vanilla | | | | | | | |
| TA | | | | | | | |
| Relay | | | | | | | |
| Repair-only | | | | | | | |
| RvI | | | | | | | |
| A2 shuffled | | | | | | | |

Official scores and auxiliary rubric statistics are never merged into one number.

Below the E2 table report the preregistered difference-in-differences `[(RvI-repair)_GLOBAL_REVISION]-[(RvI-repair)_INSERTABLE]` and its Holm-adjusted prompt-clustered CI. Full is primary; Hard is an overlapping subset and is never pooled with Full.
