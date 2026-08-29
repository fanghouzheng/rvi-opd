# Primary references and pinned upstreams

All links below are papers, official project repositories, official model pages or official dataset/evaluation sources. The related-work and comparison cutoff is **2026-08-01**; work after that date is concurrent and is neither cited nor used for head-to-head claims. Revisions were re-verified on 2026-08-30 (a retrieval date, not a literature cutoff) and are recorded in [`upstreams.lock.json`](../upstreams.lock.json).

## Method papers

- [Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation](https://arxiv.org/abs/2605.26844) and [official TA-OPD repository](https://github.com/wyy-code/TA-OPD).
- [Trajectory-Refined Distillation](https://arxiv.org/abs/2606.08432) and [official TRD repository](https://github.com/louieworth/trd).
- [SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation](https://arxiv.org/abs/2606.19659). No compatible pinned implementation is claimed; use citation/clean-room status unless an adapter is separately frozen.
- [DOPD: Dual On-Policy Distillation](https://arxiv.org/abs/2606.30626). Implementation status is audited separately from the mathematical comparison.
- [Pass the Baton: Trajectory-Relayed On-Policy Distillation](https://arxiv.org/abs/2607.26057) and [official Relay-OPD repository](https://github.com/ZJU-REAL/Relay-OPD).
- [DASH-OPD v1: Discrepancy-Aware Switching with Hysteresis](https://arxiv.org/abs/2607.29078). Only the version available by 2026-08-01 is in scope; later revisions are concurrent.
- [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) and [authors' shared OPSD repository](https://github.com/HJSang/OPSD_OnPolicyDistillation).
- [PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178) and the same [shared repository](https://github.com/HJSang/OPSD_OnPolicyDistillation).
- [Demystifying On-Policy Distillation: Length Inflation and Stabilization Strategies (Luo et al.)](https://arxiv.org/abs/2604.08527). Used for repetition/length-degradation motivation, not as proof of the RvI mechanism.
- [Speculative Knowledge Distillation: Bridging the Teacher–Student Gap Through Interleaved Sampling](https://arxiv.org/abs/2410.11325), [Google Research page](https://research.google/pubs/speculative-knowledge-distillation-bridging-the-teacher-student-gap-through-interleaved-sampling/) and [OpenReview record](https://openreview.net/forum?id=EgJhwYR2tB). The repository uses a clean-room adaptation unless an exact compatible upstream revision is frozen.
- [AdaSwitch: Adaptive Switching Generation for Knowledge Distillation](https://arxiv.org/abs/2510.07842) and [OpenReview record](https://openreview.net/forum?id=vZyrTu4dfF).
- [Fast and Effective On-Policy Distillation from Reasoning Prefixes](https://aclanthology.org/2026.findings-acl.1276/). No independent official implementation was located; Relay includes its own fixed-prefix reproduction scripts.
- [Less is More: Early Stopping Rollout for On-Policy Distillation](https://arxiv.org/abs/2605.27028). Treat ESR as a separate paper but do not claim a separately reproduced official implementation without one.

## Infrastructure and evaluation

- [verl official repository](https://github.com/verl-project/verl) and [HybridFlow paper](https://arxiv.org/abs/2409.19256).
- [DAPO project](https://dapo-sia.github.io/), [official repository](https://github.com/BytedTsinghua-SIA/DAPO) and [official DAPO-Math-17K dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). The dataset history documents the current upload lineage; physical rows must be deduplicated by prompt rather than treated as independent questions.
- [OpenAI HealthBench overview](https://openai.com/index/healthbench/), [paper](https://arxiv.org/abs/2505.08775), [official dataset](https://huggingface.co/datasets/openai/healthbench) and [official simple-evals implementation](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py). The primary protocol pins that commit's physician path (`gpt-4.1-2025-04-14`, grader max tokens 2,048, repeats 1).
- [FreedomIntelligence medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) at revision `fc2c9e8a37b38f38da6d449564a8c350b244aef4`; confirmatory E2 uses only English `medical_o1_sft.json`.
- [avaliev/chat_doctor](https://huggingface.co/datasets/avaliev/chat_doctor) at revision `19646f30de72c3890c6e0bc67579cbb538076822`; this is a post-success robustness source, not part of the confirmatory primary mixture.

## Models

- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Non-thinking is a chat-template setting, not a separate official checkpoint.
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). E2 uses the pinned post-trained student snapshot and records its `Qwen3-0.6B-Base` lineage; it is not relabelled as an Instruct checkpoint.
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) is optional robustness only. The former Qwen3-32B/4B medical pair is legacy and is not part of the final matrix.

## Baseline naming rules

- `TRD-canonical`: upstream TRD full-vocabulary FKL.
- `TRD-Relay-top128`: Relay's teacher-top-128 renormalized FKL reproduction. These are not silently merged.
- `FastOPD-Relay`: Relay's fixed prefix-length reproduction; it is not labeled an official FastOPD code release.
- TA-OPD selects by high `D^L`; describe `D^I` as reducing relative teachability, not as an explicit standalone “discard every D^I token” action in the original method.
- FastOPD/ESR alter rollout horizon. They are “fixed-horizon/truncation prevention” baselines, not literal `d=0` actions.

## Licensing note

At verification time, TA-OPD and the TIP/PACED shared repository did not expose a clear top-level license. This repository cites their formulas and uses a clean-room implementation rather than copying source. Check again before any future code import.

The two medical dataset cards above mark Apache-2.0, but card metadata alone does not supersede original-source terms. Trace provenance and document the license/consent basis before training.
