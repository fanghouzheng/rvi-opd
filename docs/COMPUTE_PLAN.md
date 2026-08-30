# Compute and execution plan

## Gate expensive work with evidence

On the `healthbench-first` branch, do not launch any mathematical run first. Execute in the following resource gates; the machine-readable authority is `configs/execution/healthbench-first.json`:

| Gate | Runs | Purpose | Stop condition |
|---|---:|---|---|
| H0 | CPU CI and E2 C0 | software/model compatibility and HealthBench leakage isolation | any C0 audit failure |
| H1 | W0 blinded rubric annotation/adjudication | freeze the output-independent rubric manifest | any W0 gate failure |
| H2 | one reference/Base resource pilot + medical D1/D2 | freeze resource, grader, threshold and manipulation-check contracts | any preregistered prerequisite failure |
| H3 | 7 E2 core trained arms × 3 seeds | acquire the complete HealthBench-first mechanism matrix without score peeking | incomplete run/hash/failure ledger |
| H4 | 25 frozen evaluation manifests on Full; Hard indexed from Full | one final confirmatory look and GO/STOP calculation | any gate check fails → `STOP_AFTER_HEALTHBENCH` |
| M0 | math D1/D2, D0/D3/A2, E1 and math ablations | mathematical-domain evidence | forbidden unless H4 returns `GO_MATH` |
| M1 | remaining E2 rows and optional robustness | complete/extend the medical table | only after H4; optional rows never rescue a failed gate |

H4 has an independent frozen-Base floor: on HealthBench Full official score, the paired RvI-minus-Base estimate must be at least `+0.01` and its seed→prompt bootstrap lower 95% bound must exceed zero. Beating all trained baselines does not release math if this floor fails; Hard remains secondary.

## Run counts

The full E1 plan is intentionally expensive:

- five trained core rows × five seeds = 25 training runs; Base and Teacher are evaluation-only rows, not seeded training runs;
- five mechanism rows × three seeds = 15 runs;
- six secondary rows × at least three seeds = 18+ runs;
- D0/D2 probes and budget sweeps are additional.

These are training-run counts, not automatic reruns. If a mechanism checkpoint has the identical resolved config, data manifest and seed required by E1, carry its immutable checkpoint/artifacts forward and count it once; a changed manifest or hyperparameter is a new named run. Evaluation-only Base/Teacher rows still incur generation/evaluation cost and are listed separately from training counts.

E2 should not mirror the whole matrix before the W0 annotation freeze and the subsequent 500-prompt reference/Base resource pilot. Stage H3 contains exactly Vanilla, Relay, canonical full-vocabulary TRD, repair-only, intervene-only, RvI and A2 at seeds `{13,17,23}`: **21 training runs**. H4 contains 25 evaluation manifests: those seven arms and Base at three sampling seeds, plus Teacher upper bound at seed 13. Only after a GO may the remaining SFT, FastOPD, SKD, TA and TIP-select rows be added. The optional 8B student and 50:50 ChatDoctor mixture are robustness-only. Keep a machine-readable run registry so failed and replacement runs cannot disappear.

## Locked medical model and prompt contract

The E2 teacher is `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` and the student is `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. The latter is a post-trained checkpoint whose lineage points to `Qwen/Qwen3-0.6B-Base`; it must not be relabelled as the `-Base` checkpoint or as an `-Instruct` model. Both use `Qwen3ForCausalLM` and have equal effective tokenizer/vocabulary content, but their tokenizer configuration, merge-file metadata and chat templates differ. The lock must therefore record per-file SHA-256 values, the model architecture metadata, and the resolved Transformers/Relay stack.

Every arm uses the project-owned `rvi_opd_non_thinking_v1` serializer. For E2 the system message is exactly `You are a helpful medical assistant. Provide a clear, complete, and safe answer while stating uncertainty and escalation conditions when relevant.`; include this text in the serializer/rendered-prompt hash (math tracks retain their own separately frozen system message). A C0 fixture renders the same synthetic conversation with both tokenizers and compares the complete token-ID sequence, special-token map, and rendered-prompt hash. Do not pass the student's official `enable_thinking=false` template option as a substitute: that template inserts an empty `<think>...</think>` block and changes the prompt IDs, whereas the teacher's template does not. The serializer, target format (`Complex_CoT`/`Response` mapping), and any `<think>` parsing rule are frozen before W0.

## Hardware starting point

Use the pinned Relay environment as the feasibility baseline. A practical starting allocation for the Qwen3-4B→1.7B long-context math setup is one 8-GPU 80GB+ node with the same tensor/FSDP/sequence parallel choices as the upstream reproduction. This is a starting point, not a promised fit: run C0 with the exact 16k response and full/gathered logit path before reserving a cluster.

The 4B→0.6B medical setup is much smaller than the original 32B→4B plan, but the teacher and student still have different context limits and tokenizer templates. A single GPU may be sufficient for inference of the 0.6B student, while teacher scoring, optimizer state, long-context KV cache and FSDP replication can require a separate teacher service or multiple GPUs. Treat this as a measured feasibility question, not a promise. The scheduler must measure teacher and student GPU-seconds independently; wall-clock alone confounds idle and parallel service time.

For the medical run, the student `max_position_embeddings` is 40,960 at the pinned revision, while [Qwen's deployment documentation](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md) documents 32,768 as the pretraining context and describes 40,960 as a runtime allocation (typically 8,192 prompt + 32,768 output). If the experiment uses the 40,960 runtime ceiling, record that it is an extension beyond the validated pretraining length and run a per-length C0 smoke. In all cases, the preflight check is on rendered IDs:

```text
rendered_input_tokens + requested_new_tokens + eos_reserve <= runtime_context_limit
```

In the declared `max_new_tokens` API, a generated stop token is counted inside `requested_new_tokens`, so `eos_reserve=0`; a backend that requires a separate reserve must record its positive value and lower the response cap before dispatch.

An over-limit request is a failed request recorded in the ledger. It must not be silently truncated or clamped by the upstream rollout server (which otherwise has a 32,768 default and may slice/clamp sequences).

## Sampling and HealthBench budget contract

Do not inherit either model's `generation_config.json`. For every local completion, pass the frozen values `do_sample=true`, `temperature=1.0`, `top_p=1.0`, an explicitly disabled `top_k`, the declared stop IDs `[151645, 151643]`, `pad_token_id=151643`, and one completion per prompt. Project configs encode disabled top-k as `0`; the Relay/verl adapter resolves this to backend value `-1` and records both. The answer-generation caps are `max_input_tokens=36,864`, `max_response_tokens=4,096`, and `max_model_length=40,960` only when the context preflight above passes; otherwise reduce the caps or mark the prompt out of contract. Keep the same prompt and sampling manifest across arms and seeds.

Freeze the HealthBench 500-prompt rubric sample, blinded labels, adjudication and annotation hash before any completion is generated. The first model-output pass is then a non-confirmatory reference/Base resource pilot on that frozen sample; it measures prompt lengths, generated tokens, grading-call counts, latency and peak memory and checks grader repeatability. Only after settings, serializer hashes and hard budgets are also frozen does the final pass generate one completion for all 5,000 Full prompts; Hard is indexed from those completions. Reserve grader budget as `sum(rubric_items_per_prompt) * repeats` rather than as a prompt count, and include the three-repeat audit on the preregistered 10% sample. A cap reached mid-matrix stops new runs with `NOT_RUN_BUDGET_CAP`; it never selects a replacement based on observed scores.

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

Before H1, freeze hard caps for annotation/adjudication and the H2 resource pilot. Immediately after that output-independent pilot and before H3, freeze `compute_budget.json` with per-gate and total caps for teacher GPU-hours, student GPU-hours, HealthBench grader calls/cost, human hours, storage and wall-clock, plus the pilot snapshot and equation used. No gate starts without its cap. When a cap is reached, finish only the current atomic run, mark the remaining matrix `NOT_RUN_BUDGET_CAP`, and report the incomplete matrix; never choose replacements using observed scientific outcomes. Mechanism tables may claim ≤1% matching only on target/scored positions, supervised tokens and optimizer steps. No table may claim strict compute matching unless measured GPU-seconds are themselves controlled by design.

After a GO, preserve the protected mathematical core: D0, D2, D3, E1, A1 and A2. Apply the attachment's cut-first order exactly: optional medical 8B/ChatDoctor robustness, then A6, then the A4 TIP-style control, then D5. Other A4–A8 extensions may be deferred only after recording the protected core and this ordering. This is an execution deferral, not permission to change hypotheses after observing scores. A STOP uses the distinct status `NOT_RUN_HEALTHBENCH_GATE` for every mathematical target; it is not a budget cut and cannot be reversed by optional E2 rows.

## Storage

Keep configs, model/tokenizer/template/sampling hashes, ID manifests, ledgers, per-problem metrics and small synthetic fixtures in Git. Store logits, trajectories, checkpoints and full HealthBench outputs in access-controlled object storage with content hashes. Cache HealthBench grading by `(prompt_id, completion_hash, rubric_id, grader_model, grader_revision, grader_template_hash, repeat_index)` without exposing item text in this repository. The 500-prompt rubric manifest and Full→Hard index manifest are metadata-only artifacts; never commit benchmark prompts, rubrics, completions or grader rationales.
