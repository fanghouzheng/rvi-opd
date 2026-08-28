# Design decisions and resolved ambiguities

## D0 includes s2 explicitly

The original 2×2 `D^L/D^I × action` cannot establish a two-signal state-dependent router because s2 is absent from the randomization. The confirmatory design is therefore 2×2×2 with an explicit s2 band. A compact paper figure may show selected contrasts, but all eight cells remain preregistered and public.

The causal mechanism estimand uses isolated frozen-prefix copies with forced randomized actions, no gate and no shared cross-state update. Full policy training is a separate independent-run check. Mixing online parameter updates into state-level randomization would create interference and pseudoreplication.

## D1 routing anchors are global and frozen

TA-style batch q05/q95 normalization remains a state descriptor/reproduction diagnostic. Router decisions instead use one raw-D/raw-C q05/q95 transform frozen globally on D1; inference batch composition and realized/future response length never change a threshold. Known prefix position is a stability diagnostic/covariate only.

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
