# Compute and execution plan

## Gate expensive work with evidence

Do not launch the full seed matrix first. Execute in resource gates:

| Gate | Runs | Purpose | Stop condition |
|---|---:|---|---|
| C0 | CPU CI + one tiny frozen-logit GPU fixture | software and distributed compatibility | any audit failure |
| C1 | D1 + D2 pilot | calibrate thresholds, H, gate null and variance | W1 fails |
| C2 | D0/D3/A2 at 3 seeds | identify action-state interaction and context cause | H1/H3/H4 fail |
| C3 | E1 core arms | external mathematical behavior | no gain over preregistered Relay comparator |
| C4 | remaining E1 baselines/ablations | full paper matrix | only after C3 |
| C5 | E2 W0 + 4B core arms | external-domain medical-question behavior | W0 or core interaction fails |
| C6 | E2 robustness / D4 / D5 | extensions | optional |

## Run counts

The full E1 plan is intentionally expensive:

- five trained core rows × five seeds = 25 training runs; Base and Teacher are evaluation-only rows, not seeded training runs;
- five mechanism rows × three seeds = 15 runs;
- six secondary rows × at least three seeds = 18+ runs;
- D0/D2 probes and budget sweeps are additional.

These are training-run counts, not automatic reruns. If a C2 checkpoint has the identical resolved config, data manifest and seed required by E1, carry its immutable checkpoint/artifacts forward and count it once; a changed manifest or hyperparameter is a new named run. Evaluation-only Base/Teacher rows still incur generation/evaluation cost and are listed separately from training counts.

E2 should not mirror the whole matrix before W0 and the 4B core comparison pass. Stage C5 starts with Base evaluation plus Vanilla, Relay, RvI and A2 at three training seeds (12 training runs); only after that gate add SFT, TA, repair-only, intervene-only and TRD. The 8B student and 50:50 ChatDoctor mixture are robustness-only. Keep a machine-readable run registry so failed and replacement runs cannot disappear.

## Hardware starting point

Use the pinned Relay environment as the feasibility baseline. A practical starting allocation for the Qwen3-4B→1.7B long-context math setup is one 8-GPU 80GB+ node with the same tensor/FSDP/sequence parallel choices as the upstream reproduction. This is a starting point, not a promised fit: run C0 with the exact 16k response and full/gathered logit path before reserving a cluster.

The 32B→4B medical setup will generally need more teacher inference memory or a separated teacher service. The scheduler must measure teacher and student GPU-seconds independently; wall-clock alone confounds idle and parallel service time.

## Pilot-based estimate

For each representative arm, measure after warm-up:

```text
prompt_count
student_rollout_tokens
teacher_prefill_tokens
teacher_scored_tokens
teacher_generated_tokens
gate_teacher_scored_tokens
optimizer_supervised_tokens
teacher_gpu_seconds
student_gpu_seconds
wall_seconds
peak_memory_gib_per_rank
```

Estimate each workload class separately; a single throughput applied to a sum of prefill, teacher scoring, autoregressive decode and optimizer tokens is invalid:

```text
teacher_gpu_h = (prefill_tokens / prefill_tok_s
               + score_positions / score_pos_s
               + autoregressive_decode_tokens / decode_tok_s
               + gate_score_positions / gate_score_pos_s) / 3600
student_gpu_h = (rollout_tokens / rollout_tok_s
               + supervised_tokens / optimizer_tok_s) / 3600
```

Prefer an empirical end-to-end regression per arm when batching/parallel services make those terms non-additive. `teacher_forward_calls` is diagnostic only: dynamic batches make a call neither a fixed token count nor a fixed FLOP unit. Use p50 for capacity planning and p90 for reservation buffers. Publish predicted versus realized cost per arm; if the error is large, update only the compute plan, not scientific thresholds.

Before C1, freeze `compute_budget.json` with per-gate and total hard caps for teacher GPU-hours, student GPU-hours, HealthBench grader calls/cost, human annotation/adjudication hours, storage and wall-clock, plus the pilot snapshot and equation used. No gate starts without a cap. When a cap is reached, finish only the current atomic run, mark the remaining matrix `NOT_RUN_BUDGET_CAP`, and report the incomplete matrix; never choose replacements using observed scientific outcomes. Mechanism tables may claim ≤1% matching only on target/scored positions, supervised tokens and optimizer steps. No table may claim strict compute matching unless measured GPU-seconds are themselves controlled by design.

## Storage

Keep configs, ID/hash manifests, ledgers, per-problem metrics and small fixtures in Git. Store logits, trajectories, checkpoints and full HealthBench outputs in access-controlled object storage with content hashes. Cache HealthBench grading by `(prompt_id, completion_hash, rubric_id, grader_revision)` without exposing item text in this repository.
