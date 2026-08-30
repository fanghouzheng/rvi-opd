# Reproducibility, leakage and artifact policy

## Immutable inputs

Before the first GPU run, replace every mutable model/data name with a commit SHA and save:

- model and tokenizer revisions plus per-file hashes, vocabulary hash, architecture metadata and model lineage;
- dataset revision and pre/post-dedup row counts;
- upstream repository commit SHAs;
- CUDA, driver, PyTorch, vLLM/SGLang and verl versions;
- container image digest and hardware topology;
- exact project-owned chat serializer, rendered-prompt fixture/hash, stop tokens and generation config (including explicit sampling overrides).

The repository intentionally keeps names in readable experiment configs and records verified SHAs in `upstreams.lock.json`. For E2, the immutable pair is `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` (teacher) and `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` (student). The student is a post-trained checkpoint whose lineage points to `Qwen/Qwen3-0.6B-Base`; do not silently substitute the `-Base` or an `-Instruct`-named checkpoint. A run is invalid if its resolved manifest still contains `main`, `latest` or a floating model revision.

Environment creation and verification must follow [`ENVIRONMENT.zh-CN.md`](ENVIRONMENT.zh-CN.md). Confirmatory runs use the pinned Relay environment as one atomic lock; installing standalone verl/TRD or independently upgrading Torch, Transformers, vLLM, Triton or NumPy invalidates the environment fingerprint.

The two checkpoints share effective tokenizer vocabulary files but not tokenizer configuration or chat template. Freeze `rvi_opd_non_thinking_v1` as the canonical serializer; for E2 its system message is exactly `You are a helpful medical assistant. Provide a clear, complete, and safe answer while stating uncertainty and escalation conditions when relevant.` and is included in the rendered-prompt hash. Compare complete rendered token-ID sequences in C0, and record special-token IDs, `config.vocab_size` and tokenizer length separately. Passing `enable_thinking=false` to only the student is not an equivalent serializer because it inserts an empty `<think>...</think>` block.

Production routing loads a frozen threshold artifact by content hash; it must not require the calibration rows or silently recalibrate. Every route event stores `threshold_artifact_sha256`. `route_events.jsonl` is explicitly pre-gate and contains only `requested_action`; it must not invent an `effective_action`. The executor writes `action_events.jsonl` after the frozen joint gate, with requested/effective actions, gate status/hash, rollback reason, payload hash, realized length and cost signature.

## Data splitting and decontamination

### Math

1. Normalize Unicode, whitespace, LaTeX, answer formatting and numeric literals.
2. Exact-hash normalized prompts.
3. Cluster word/character n-gram MinHash candidates.
4. Compare equation signatures and distinctive phrase matches.
5. Manually review high-similarity candidates.
6. Split by the resulting cluster, never physical row.

The pinned `DAPO-Math-17K@65877096c24ffa7abc4e4fa5edb95cf3413a5674` snapshot contains 1,791,700 physical rows despite the dataset name; always publish raw, English-filtered, unique normalized-cluster, post-benchmark-denylist, train, calibration and diagnostic counts plus the cluster/final-manifest hashes. Relay's exact English-subset ID manifest was not public at verification time, so disclose any dataset-manifest mismatch rather than calling it exact.

### Medical

The confirmatory source is only `FreedomIntelligence/medical-o1-reasoning-SFT@fc2c9e8a37b38f38da6d449564a8c350b244aef4`, English `medical_o1_sft.json` (SHA256 `6a0289ebe0d07e3e77e8dbc81c31fcfc02cc99c7bca1602694cb2f76d505667e`), split 80/10/10 by normalized/near-duplicate cluster. Validate its `Question/Complex_CoT/Response` schema before mapping. `avaliev/chat_doctor@19646f30de72c3890c6e0bc67579cbb538076822` enters only a post-success, 50:50 source-balanced robustness run using `train.json` and its `instruction/input/output` schema; re-cluster the combined pool and exclude its source test/validation files. Freeze raw/source/unique/post-denylist/split counts and manifest hashes. Both HF cards mark Apache-2.0, but that metadata does not override original data terms: a provenance/license audit is a run gate.

HealthBench prompts, rubrics, examples, references and grader rationales are all denylisted. Perform exact and semantic matching against every training source before D1, gate or training. The data snapshot is `openai/healthbench@40ee1968852fc57f625934251ac22be47077a8fb`; record the Full/Hard JSONL file hashes (currently Full `e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f`, Hard `b0320430e5cd974e746585594c1dd10b5a3fc2aff9c72b26106c2c4a069d74e9`), and freeze the Hard→Full mapping as compact UTF-8 JSONL (`{prompt_id,full_index}` in Hard-file order, LF) with SHA256 `64852846390fa7b3f65e1f0ae93d0160318188b6264023673184eef2dcf7bca7`. Verify unique prompt IDs with Hard a subset of Full before generating anything. Full and Hard share examples and are evaluated/reported separately rather than pooled as independent observations. The auxiliary rubric-type analysis uses a **500-prompt** sample (a reduction to 300 is allowed only if frozen before any model output), stratified by Hard membership and frozen official theme; freeze its seed/manifest, complete blinded labeling/adjudication, and hash every included rubric item before generating the resource-pilot completion. This is the W0 annotation gate, distinct from the later non-confirmatory model-output resource pilot. Use `openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47`, grader `gpt-4.1-2025-04-14`, grader max tokens 2,048 and primary repeats=1; record the unmodified physician sampler/template hash. The simple-evals CLI does not generate completions from local Qwen checkpoints, so keep local generation and official grading as separate, hashed stages.

For every final completion, explicitly set `do_sample=true`, `temperature=1.0`, `top_p=1.0`, disable `top_k`, use stop IDs `[151645, 151643]` and `pad_token_id=151643`, and reuse one prompt/sampling manifest across arms. The project config represents disabled top-k as `0`; the Relay/verl adapter must translate it to backend value `-1` and record the resolved request. Do not inherit model-card generation defaults. Check `rendered_input_tokens + requested_new_tokens + eos_reserve <= runtime_context_limit` before calling the rollout backend; `eos_reserve=0` when the backend counts a stop token inside `max_new_tokens`, otherwise record the reserve and lower the response cap. An over-limit request is recorded as a failure, never silently truncated or clamped. The E2 runtime caps are input 36,864, response 4,096 and model length 40,960; Qwen's documented pretraining context is 32,768, so inputs beyond that boundary require an explicit long-context extension note and C0 length-stratified smoke.

Foundation-model pretraining contamination cannot be removed; disclose it. All trained E2 arms start from the same locked `Qwen/Qwen3-0.6B` student checkpoint (the teacher is a separate locked checkpoint used for scoring/trajectories), so do not describe this as a comparison of independently pretrained students.

## Benchmark access discipline

- D1/D2 use training-derived holdouts, never final benchmarks.
- Choose H, K, global raw-D/C anchors, thresholds, gate, loss support and ablations before final benchmark access. Batch q05/q95 values are descriptors only and never replace the frozen global transform; no threshold uses realized/future response length.
- First freeze the 500-prompt rubric sample, complete blinded labeling/adjudication and hash the annotation artifact before any model completion is generated. Then run one locked reference/Base checkpoint on that frozen sample as a resource/grader pilot for length, variance, repeatability and resource estimates; its completions are excluded from arm comparisons. Freeze the model/revision pair, serializer/template hashes, generation settings and hard budget before final access.
- Run final Full generation once per preregistered seed manifest (5,000 prompts); derive Hard by indexing those same completions by prompt ID, not by a second sampling pass.
- Any rerun caused by infrastructure failure keeps the same seed and is recorded as a failed/replaced run.

On the `healthbench-first` branch, the eight math/mechanism configs are frozen before any final HealthBench output is visible and bound as one ordered `math_config_bundle_sha256`. HealthBench has one preregistered Full look and produces an append-only gate evidence artifact; it may not select a math threshold, hyperparameter, seed, baseline or ablation. One necessary gate is the paired RvI-minus-frozen-Base Full official-score delta: estimate at least `+0.01` and seed→prompt bootstrap lower 95% bound above zero. Only a recomputed `GO_MATH` for this and every other check, under the exact policy/code/E2-config/math-bundle hashes, releases math. A STOP records every math target as `NOT_RUN_HEALTHBENCH_GATE`.

## Run artifact contract

```text
runs/<run_id>/
  resolved_config.json
  manifest.json
  threshold_artifact.json
  joint_gate_artifact.json
  c0_artifact.json
  prompt_manifest.jsonl
  states.jsonl
  route_events.jsonl
  action_events.jsonl
  budget_ledger.jsonl
  continuations.jsonl
  per_problem_metrics.jsonl
  metrics.json
  audit.json
  environment.json
  completion_manifest.jsonl
  healthbench_index_manifest.json
  _SUCCESS

runs/gates/
  healthbench-first.json
  math-release.json
```

Large trajectories/checkpoints and HealthBench prompt/completion/grader payloads should live in controlled object storage; Git stores schemas, small synthetic fixtures and content hashes only. `completion_manifest.jsonl` and `healthbench_index_manifest.json` contain metadata/IDs/hashes, not benchmark content. Write artifacts atomically, and create `_SUCCESS` only after all hashes and invariants pass.

`runs/gates/healthbench-first.json` is append-only evidence, not a hand-authored decision. It binds the execution-policy SHA256, 40-hex code revision, E2-config SHA256, ordered math-config bundle SHA256, all 21 training run/checkpoint hashes and all 25 completion/grader manifest hashes. The evaluator recomputes GO/STOP in a clean checkout and writes `math-release.json`; every released math run records the source gate evidence SHA256. Changing any bound config or code invalidates the old release. As elsewhere under `runs/`, gate storage contains hashes and aggregate statistics only—never HealthBench content.

`run_id` combines git SHA, resolved-config hash and seed. `state_id` combines problem ID, rollout seed, token index and tokenizer hash.

## Required audits

- schema and content hashes;
- model/tokenizer/vocabulary equality where required, plus canonical rendered-prompt equality;
- explicit sampling parameters, stop/pad IDs and context-budget arithmetic;
- no benchmark denylist match;
- paired-state and paired-seed completeness;
- all declared budget axes within tolerance;
- detached context equality and normal context difference;
- gate rollback restores the exact original context;
- padding/EOS/loss-mask correctness;
- no secret, model weight, benchmark sample or private path in Git.

Every gate has a frozen hard budget for teacher GPU-hours, student GPU-hours, optimizer/rollout tokens, HealthBench grader calls and API cost, annotation hours, storage and wall-clock. Count grader calls per rubric item and repeat (including retries and the 10% three-repeat audit), not merely per prompt. On exhaustion, finish only the current atomic request, mark unstarted rows `NOT_RUN_BUDGET_CAP`, and never choose replacements from observed scores.

## Statistical reproducibility

Publish raw per-problem/per-rubric scores, bootstrap seed and resampling code. Problem/prompt—not token, state or rollout—is the cluster. Training seeds and sampling rollouts form separate levels in E1. With only three D0 policy seeds, analyze seed-level fixed contrasts rather than an unstable random-effect variance. D2 binary outcomes and D5 arm comparisons use prompt-cluster paired bootstrap/GEE, not simple McNemar or per-prompt Clopper–Pearson inference. Confirmatory p-values use Holm; exploratory ablations use BH-FDR.

## Baseline fairness

Distinguish an upstream method from a reimplementation choice. In particular, record whether TRD uses full-vocabulary FKL or Relay's top-128 renormalized adaptation. Relay reproduction must preserve K=5, M=2, L=3, cap=256, no cooldown, M-th-leg termination and actual-token k1 RKL. A changed cooldown/cap is an RvI variant. All head-to-head rows use the same prompt formatting, generation limits, dataset manifest and evaluation harness unless the method definition makes that impossible; every exception is a named column in the main table. Freeze cross-benchmark problem IDs and near-duplicate clusters, and label k-sample accuracy `avg@k`, not `pass@k`.
