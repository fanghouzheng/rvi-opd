# Method and implementation contract

## 1. State signals

For a student context `c_t`, define student and teacher top-K supports `S^S_t(K)` and `S^T_t(K)`, and `U_t = S^S_t ∪ S^T_t`. Teacher and student probabilities must come from the same context, tokenizer and vocabulary.

```text
D_t = KL(p_T^U || p_S^U)
C_t = sum_{v in S^S_t(K)} p_T(v | c_t)
```

Each distribution is independently renormalized on `U_t`. To reproduce TA-OPD diagnostics, within a rollout batch one may report:

```text
Norm(z) = clip((z - Q05(z)) / (Q95(z) - Q05(z) + eps), 0, 1)
D^L = D_tilde * C_tilde
D^I = D_tilde * (1 - C_tilde)
```

Those batch q05/q95 values describe states **within that batch only**; they are not portable decision thresholds. Production/replay routing loads one global pair of raw-`D` and raw-`C` q05/q95 anchors frozen on the D1 calibration split and applies that fixed transform to every later state, independent of inference-batch composition. `route-jsonl` binds the four anchors, quantile levels, vocabulary SHA-256 and code revision into a frozen-scale hash stored by the threshold artifact; replay reconstructs and verifies that scale solely from the artifact. Batch-wise decomposition remains a TA diagnostic and is never used by that routing command. Known prefix position `token_index/max_response_tokens` is reported in four bins as a stability diagnostic/covariate, not as multiple primary router thresholds. Realized or future response length is forbidden. Define `s1=max(D^L,D^I)` after the fixed transform; `s1` is disagreement eligibility, not literally an absorbability probability.

The code implements the teacher-weighted forward direction `KL(T||S)`. Missing union logits are a data error: do not concatenate two truncated probability arrays and silently treat missing tokens as exact zero.

## 2. Prefix-damage signal

`s2` is untempered teacher probability mass on tokenizer-specific first-subword IDs derived from the 16 onset phrases used by TRD:

```text
Wait, Actually, However, Alternatively, Oops, Wrong, Error, Incorrect,
Correction, Sorry, Hmm, Oh, Hold, Pause, Uh, Um
```

For every phrase, tokenize bare and leading-space variants, take the first subword ID, then deduplicate. Save the actual token sequences, selected IDs, tokenizer revision/hash and vocabulary hash. Numeric thresholds are calibrated per model/domain; values reported by another paper are not defaults.

This set is **not** Relay's handoff lexicon. Relay uses 13 bases—`Wait, But, Hmm, Actually, Hold, However, Yet, Oh, Alternatively, No, Ah, Oops, Well`—with case and leading-space variants. Keep only variants whose entire tokenizer encoding has exactly one token; never coerce a multi-token phrase to its first subtoken. Code and artifacts name these separately as `EPISTEMIC_ONSET_PHRASES` and `RELAY_REFLECTION_PHRASES`.

## 3. Frozen core router

```text
if optional_repetition_bypass:
    teacher_has_alternative -> intervene
    otherwise -> discard
elif optional_paced_zero_rescue and p_hat == 0:
    intervene
elif s2 >= tau2 and phi(c) == 1 and budget/cooldown permits:
    propose intervene
    paired gate accepts -> intervene
    otherwise -> repair
elif max(DL, DI) >= tau1:
    repair
else:
    discard
```

D4 and D5 branches are disabled in the core D0–D3 confirmatory router. `route → budget reservation → action → ledger append` are separate stages; the router never spends compute itself.

## 4. Actions

### Repair (`d=0`)

At the original context, minimize teacher cross-entropy over teacher-top-128 support:

```text
L_FCE = -sum_v q_T^K(v|c_t) log p_S(v|c_t)
```

Here `q_T^K` is renormalized on teacher top-128, while `log p_S(v|c_t)` comes from the student's **full-vocabulary softmax**. Under that exact convention, FCE and FKL have the same student gradient because teacher entropy is constant; they are not separate ablation arms. Renormalizing the student again on the support changes the objective and must be named separately. A6 varies support and KL direction, not additive teacher-entropy constants.

The D0/D2 causal probe uses a fresh zero-effect LoRA per isolated state: rank 8, alpha 16, dropout 0, target `q/k/v/o/gate/up/down_proj`; AdamW LR `1e-4`, betas `0.9/0.95`, eps `1e-8`, weight decay 0, clip 1, FP32 optimizer state. Take at least one step and stop at 20% relative marked-position KL reduction or 8 steps. Max-step failures remain in ITT. Marked-position KL is a manipulation check because it also defines the update/stop rule.

### Intervene (`d=L`)

Let `R` be the frozen set of single-token Relay IDs and `TopK_S(c;5)` the student's top five IDs. Relay eligibility is exactly

```text
phi(c) = 1[argmax_v p_T(v|c) in R] * 1[TopK_S(c;5) intersect R is empty].
```

The teacher leg begins with that teacher global-argmax token, then generates up to `L=3` complete `\n\n`-delimited paragraphs, with a 256-token hard cap. High s2 alone is not `phi=1`. The canonical Relay baseline is K=5, M=2, L=3, cap=256, no cooldown, termination after the M-th leg, and k1 RKL on the actual emitted relay token. RvI policy training separately freezes cooldown=1; any other cap/resume rule is labeled an RvI choice, not Relay reproduction. D0's isolated forced-action probe and D3 use the first eligible trigger only (`M=1`) and force a student resume so a post-leg outcome exists. Actual generated tokens, not paragraph counts, define cost.

### Detached

Generate/score the exact same teacher leg from the original prefix and apply the same Relay k1 RKL on the actual emitted relay token as normal intervention. Then delete every bridge KV, reset position IDs, and start a separate continuation pass from the original prefix. An implementation that merely stops gradients through retained bridge KV violates the contract.

## 5. Training-time acceptance gate

For each proposal, use paired frozen-student continuations from base and bridge contexts. Let:

The gate is part of the deployed/end-to-end RvI router and A4. It is disabled in randomized D0 and causal D3 so treatment assignment cannot be changed after randomization.

```text
s2_residual = mean_t sum_{r in R} p_T(r | continuation_prefix_t)
agree@K = mean_t 1[argmax(p_T) in TopK(p_S)]
```

D1 obtains one global joint null distribution from ineffective/random bridges. Orient both improvements positively, standardize them with frozen null summaries, and define `G=max(z_s2_reduction,z_agree_gain)`. Accept only if `G` exceeds the global frozen q95 of the **joint max-statistic null**. Testing each metric against its own q95 and accepting their OR inflates event-wise false acceptance and violates the contract. Known prefix position is a diagnostic/prespecified covariate, not a set of primary gate cutoffs. The dependency-free core serializes and validates the joint artifact; the GPU adapter must use the joint evaluator, not the absolute-threshold OR retained only for synthetic smoke.

If rejected, `requested_action=intervene` and `effective_action=repair`. Generated bridge and gate-scoring costs remain in the ledger. Context must roll back bit-for-bit to the original hash before repair.

## 6. Cost vector and matching

Every event records at least:

```text
examples
teacher_scored_tokens
teacher_generated_tokens
teacher_inserted_tokens
gate_teacher_scored_tokens
teacher_prefill_tokens
teacher_forward_calls
student_rollout_tokens
student_supervised_tokens
optimizer_steps
teacher_gpu_seconds
student_gpu_seconds
wall_time_ms
```

Mechanism experiments match examples, `teacher_scored_tokens`, student supervised tokens and optimizer steps. In intervene, `teacher_scored_tokens` counts each autoregressive teacher decode position; in repair it counts each teacher distribution-scoring position. It is therefore a target/query-position equivalent, not a FLOP or compute equivalence. Dynamic batching makes raw forward-call counts non-comparable, so `teacher_forward_calls`, prefill, decode, gate scoring and measured teacher/student GPU-seconds are reported but never called matched. `teacher_inserted_tokens` and `teacher_generated_tokens` remain explicit mechanism/cost descriptors and are not declared matched to repair.

For realized bridge lengths `{b_e}`, allocate `sum_e b_e` repair-scored positions from the same preregistered stratum. If the original trajectory is too short, cap the paired bridge or draw a preregistered replacement; never backfill from a different stratum after seeing outcomes. A secondary compute control may generate a bridge for repair and discard it before the repair action; match its prefill/decode tokens and measured teacher GPU-seconds, but label it a sham-generation control rather than part of deployment.

Deployment tables do not generate sham bridge tokens for repair. They hold student rollout/optimizer budgets fixed and compare task score versus measured teacher GPU-seconds.

## 7. Required action event

Each event must contain:

```text
run_id, problem_id, state_id, prefix_hash, token_index,
raw_D, raw_C, D_tilde, C_tilde, DL, DI, s2,
threshold_artifact_hash, requested_action, effective_action,
bridge_token_ids, bridge_hash, context_hash_before, context_hash_after,
gate_base_metrics, gate_bridge_metrics, gate_decision, rollback_reason,
loss_support, loss_direction, loss_mask_hash, CostVector
```

State IDs are hashes of problem ID, rollout seed, token index and tokenizer hash. A state ID must never depend on the action or outcome.

For A2, first freeze RvI's requested action and post-gate effective action plus `(bridge_token_length, payload_hash, cost_signature)` before any training/task outcome is observed. Permute that inseparable bundle within preregistered composite blocks containing problem/difficulty/position strata, and do not rerun the gate at the destination. Shuffling only the action label—or shuffling first and letting a destination-specific gate alter realized counts—does not satisfy the production A2 contract.
