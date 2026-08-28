# Reproducibility, leakage and artifact policy

## Immutable inputs

Before the first GPU run, replace every mutable model/data name with a commit SHA and save:

- model and tokenizer revisions plus vocabulary hash;
- dataset revision and pre/post-dedup row counts;
- upstream repository commit SHAs;
- CUDA, driver, PyTorch, vLLM/SGLang and verl versions;
- container image digest and hardware topology;
- exact chat template, stop tokens and generation config.

The repository intentionally keeps names in readable experiment configs and records verified SHAs in `upstreams.lock.json`. A run is invalid if its resolved manifest still contains `main`, `latest` or a floating model revision.

Production routing loads a frozen threshold artifact by content hash; it must not require the calibration rows or silently recalibrate. Every route event stores `threshold_artifact_sha256`.

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

HealthBench prompts, rubrics, examples, references and grader rationales are all denylisted. Perform exact and semantic matching against every training source before D1, gate or training. Full and Hard share examples and are evaluated/reported separately rather than pooled as independent observations. The auxiliary rubric-type analysis uses a 1,000-prompt sample stratified by Hard membership and frozen official theme; freeze its seed/manifest and label every included rubric item before model outputs. Use `openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47`, grader `gpt-4.1-2025-04-14`, grader max tokens 2,048 and primary repeats=1; record the unmodified physician sampler/template hash.

Foundation-model pretraining contamination cannot be removed; disclose it and note that all arms share the same base checkpoint.

## Benchmark access discipline

- D1/D2 use training-derived holdouts, never final benchmarks.
- Choose H, K, global raw-D/C anchors, thresholds, gate, loss support and ablations before final benchmark access. Batch q05/q95 values are descriptors only and never replace the frozen global transform; no threshold uses realized/future response length.
- Run final test generation once per preregistered seed manifest.
- Any rerun caused by infrastructure failure keeps the same seed and is recorded as a failed/replaced run.

## Run artifact contract

```text
runs/<run_id>/
  resolved_config.json
  manifest.json
  threshold_artifact.json
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
  _SUCCESS
```

Large trajectories/checkpoints should live in controlled object storage; Git stores schemas, small synthetic fixtures and content hashes only. Write artifacts atomically, and create `_SUCCESS` only after all hashes and invariants pass.

`run_id` combines git SHA, resolved-config hash and seed. `state_id` combines problem ID, rollout seed, token index and tokenizer hash.

## Required audits

- schema and content hashes;
- model/tokenizer/vocabulary equality where required;
- no benchmark denylist match;
- paired-state and paired-seed completeness;
- all declared budget axes within tolerance;
- detached context equality and normal context difference;
- gate rollback restores the exact original context;
- padding/EOS/loss-mask correctness;
- no secret, model weight, benchmark sample or private path in Git.

## Statistical reproducibility

Publish raw per-problem/per-rubric scores, bootstrap seed and resampling code. Problem/prompt—not token, state or rollout—is the cluster. Training seeds and sampling rollouts form separate levels in E1. With only three D0 policy seeds, analyze seed-level fixed contrasts rather than an unstable random-effect variance. D2 binary outcomes and D5 arm comparisons use prompt-cluster paired bootstrap/GEE, not simple McNemar or per-prompt Clopper–Pearson inference. Confirmatory p-values use Holm; exploratory ablations use BH-FDR.

## Baseline fairness

Distinguish an upstream method from a reimplementation choice. In particular, record whether TRD uses full-vocabulary FKL or Relay's top-128 renormalized adaptation. Relay reproduction must preserve K=5, M=2, L=3, cap=256, no cooldown, M-th-leg termination and actual-token k1 RKL. A changed cooldown/cap is an RvI variant. All head-to-head rows use the same prompt formatting, generation limits, dataset manifest and evaluation harness unless the method definition makes that impossible; every exception is a named column in the main table. Freeze cross-benchmark problem IDs and near-duplicate clusters, and label k-sample accuracy `avg@k`, not `pass@k`.
