# GPU upstream integration plan

## One execution stack

Use the pinned Relay-OPD subproject as the primary training stack because it already contains the Relay rollout engine, locked environment and baseline scripts. Do not compare results produced independently on current verl, Slime/Megatron and the TRD fork without first eliminating stack-level differences.

Clean-room port only the mathematical selectors/loss masks required for TA, TIP and PACED. Canonical TRD remains a separate upstream reproduction because its full-vocabulary FKL differs from Relay's top-128 adaptation.

## Adapter stages

1. `score_union`: materialize both teacher and student probabilities for every token in the top-K union and separately gather all s2 onset IDs, even outside top-K.
2. `normalize_batch`: reproduce batch q05/q95 D/C only as a descriptor; separately load D1's immutable global raw-D/raw-C q05/q95 anchors for every routing transform. Batch composition and future response length never affect a route.
3. `route_only`: produce immutable route plans without generating or updating.
4. `reserve_budget`: atomically reserve `M`, cooldown and declared cost quota.
5. `execute_action`: repair loss, Relay leg or detached leg; emit requested/effective actions and context hashes.
6. `paired_gate`: use frozen student seed pairs and append cost before accept/rollback.
7. `optimize`: consume the finalized trajectory/loss mask.
8. `audit`: refuse `_SUCCESS` if schema, pairing, context or budget invariants fail.

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
- detached context hash behavior.

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
