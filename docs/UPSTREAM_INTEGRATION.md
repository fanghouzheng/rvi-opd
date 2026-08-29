# GPU upstream integration plan

## One execution stack

Use the pinned Relay-OPD subproject as the primary training stack because it already contains the Relay rollout engine, locked environment and baseline scripts. Do not compare results produced independently on current verl, Slime/Megatron and the TRD fork without first eliminating stack-level differences.

The executable dependency contract is centralized in [`ENVIRONMENT.zh-CN.md`](ENVIRONMENT.zh-CN.md), `environment-lock.json`, and the pinned Relay installer. The standalone verl commit in `upstreams.lock.json` is reference/porting-only: its vLLM 0.24 / Transformers 5.5.3 / NumPy 2 profile is incompatible with the confirmatory Relay vLLM 0.21 / Transformers 5.14.1 / NumPy 1.26 profile and must never be installed into the same environment.

Clean-room port only the mathematical selectors/loss masks required for TA, TIP and PACED. Canonical TRD remains a separate upstream reproduction because its full-vocabulary FKL differs from Relay's top-128 adaptation.

## E2 model and environment compatibility

The medical pair is fixed to teacher `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` and student `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. Both resolve to `Qwen3ForCausalLM` without `trust_remote_code`. The student is a post-trained hybrid checkpoint derived from `Qwen3-0.6B-Base`; keep that lineage metadata separate from the checkpoint ID and do not substitute the `-Base` or an `-Instruct` variant.

The pinned Relay commit is the only training/rollout stack used for head-to-head rows. Its Qwen3 examples do not by themselves validate the 0.6B medical pair, so C0 must load both exact `AutoConfig`/`AutoTokenizer` revisions and run a one-step HF/vLLM/FSDP fixture before a GPU reservation. Record Python, PyTorch, Transformers, vLLM, verl, CUDA and driver versions in the resolved manifest. No result is portable across a changed vLLM patch or tokenizer revision.

The two tokenizers have equal effective vocabulary files and special-token IDs, but their `tokenizer_config`, `merges.txt` metadata and chat templates differ. Compare the complete token-ID sequence of a project-owned synthetic conversation, not just a model name or `vocab_size`; record `config.vocab_size`, tokenizer length and every relevant file hash.

## Adapter stages

1. `score_union`: materialize both teacher and student probabilities for every token in the top-K union and separately gather all s2 onset IDs, even outside top-K.
2. `normalize_batch`: reproduce batch q05/q95 D/C only as a descriptor; separately load D1's immutable global raw-D/raw-C q05/q95 anchors for every routing transform. Batch composition and future response length never affect a route.
3. `route_only`: produce immutable route plans without generating or updating.
4. `reserve_budget`: atomically reserve `M`, cooldown and declared cost quota.
5. `execute_action`: repair loss, Relay leg or detached leg; emit requested/effective actions and context hashes.
6. `paired_gate`: use frozen student seed pairs and append cost before accept/rollback.
7. `optimize`: consume the finalized trajectory/loss mask.
8. `audit`: refuse `_SUCCESS` if schema, pairing, context or budget invariants fail.

### Canonical serializer, sampling and context checks

All arms use the project-owned `rvi_opd_non_thinking_v1` serializer. For E2 freeze the exact system message `You are a helpful medical assistant. Provide a clear, complete, and safe answer while stating uncertainty and escalation conditions when relevant.` together with the byte representation, rendered token IDs and `Complex_CoT`/`Response` target mapping before W0; include the system text in the serializer hash. The official student template's explicit `enable_thinking=false` option inserts an empty `<think>...</think>` block and therefore is not equivalent to the teacher template; do not pass it to only one model. If a completion contains a thinking block, apply the pre-registered parser/violation policy and retain the original hash.

Generation must explicitly set `do_sample=true`, `temperature=1.0`, `top_p=1.0`, disabled `top_k`, stop IDs `[151645, 151643]`, `pad_token_id=151643` and one completion per prompt. Configs use project-level `top_k=0`; the Relay/verl adapter maps that to its documented disabled value `-1` before dispatch and records both values. Never inherit either model's `generation_config.json`. Before every call, enforce:

```text
rendered_input_tokens + requested_new_tokens + eos_reserve <= runtime_context_limit
```

For E2 the declared runtime caps are input 36,864, response 4,096 and model length 40,960. The `max_new_tokens` cap includes a generated stop token (`eos_reserve=0`); if a backend requires a separate reserve, lower the response cap and record it in the manifest. Qwen documents 32,768 as the pretraining context and 40,960 as a runtime allocation; any input over 32,768 is a separately labelled long-context extension with a length-stratified smoke. A request that exceeds the chosen limit must fail before dispatch and enter the budget ledger. Do not rely on Relay/verl's default 32,768 sequence slice or vLLM's remaining-token clamp, both of which can otherwise violate the no-silent-truncation contract.

The HealthBench adapter runs in two immutable stages: (1) generate Full completions from local checkpoints with a shared prompt/sampling manifest; (2) pass those completions to the unmodified physician grader at the locked simple-evals commit. Before any completion generation, freeze the 500-prompt rubric sample, complete blinded labeling/adjudication and hash that artifact. Only then may the locked reference/Base checkpoint produce the non-confirmatory W0 resource pilot used to estimate token/latency/memory/grader-call budgets. After all remaining hashes and hard caps are frozen, run Full once for 5,000 prompts and derive Hard by prompt-ID index (1,000 rows), never by a second sampling pass. Keep benchmark payloads and grader rationales out of Git.

## Frozen-logit compatibility fixture

Before any training, export a small synthetic/full-vocabulary logit fixture from the pinned stack and require independent implementations to match:

- top-K sets and union IDs;
- raw forward KL and teacher coverage;
- q05/q95 normalized D/C and `DL+DI=D_tilde`;
- batch-descriptor versus frozen-global replay transforms and invariance to batch regrouping;
- TRD and Relay lexicon token ID artifacts;
- Relay variants retain only whole encodings of exactly one token; exact `phi` K=5 behavior;
- repair loss and gradient on teacher-top-128;
- route decision and budget reservation;
- detached context hash behavior;
- teacher/student canonical rendered-prompt IDs, special-token map, serializer hash and loss-mask boundaries;
- explicit sampling payload (temperature, top-p, top-k disable, stop IDs, pad ID) and context-budget rejection at the boundary;
- Full/Hard prompt-ID subset and index hash (canonical compact JSONL mapping SHA256 `64852846390fa7b3f65e1f0ae93d0160318188b6264023673184eef2dcf7bca7`), with no benchmark content in the fixture.

This fixture contains no benchmark prompt or model output and can be committed if small.

## Required implementation tests beyond the CPU core

- bf16/fp32 tolerance for full-vocab versus gathered logits;
- s2 mass coverage error when using explicit gather versus full softmax;
- padding, EOS and sequence-parallel loss masks;
- same-seed paired continuation under distributed sampling;
- bridge paragraph stop and pinned Relay 256-token cap behavior;
- Relay no-cooldown and M-th takeover terminal behavior matching the pinned revision;
- actual emitted-relay-token k1 RKL, plus D3 first-trigger M=1 forced resume;
- global joint gate max-statistic q95 rather than an OR of two marginal q95 tests;
- gate rollback restores KV/cache and prefix hash exactly;
- rejected gate compute remains in the ledger;
- trainer resume does not double-spend reserved budget.
- model loading rejects an ID/revision, architecture, tokenizer hash or `config.vocab_size` mismatch;
- local completion manifests are accepted by the official simple-evals physician grader without changing its prompt, rubric, sampler or aggregation;
- a 500-prompt W0 pilot is marked non-confirmatory and its grader calls/retries are charged to the hard budget;
- Full completions are generated once and Hard is indexed from the same completion hash, never independently sampled.
