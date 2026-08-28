# RvI-OPD 完整实验计划

## 1. 研究问题与可证伪主张

核心研究问题不是“哪个 loss 更强”，而是：**在教师计算预算可比时，状态是否决定正确的处置深度？**

预注册四个确认性假设：

- H1：动作优势同时随 disagreement type 与 prefix damage 改变。主 contrast 为
  `Δ3=[(I−R)DI,high−(I−R)DI,low]−[(I−R)DL,high−(I−R)DL,low] > 0`；其中 `I=intervene`、`R=repair`，成功条件是问题聚类推断下 `Δ3` 的 95% CI 下界大于 0。
- H2：局部 repair 可降低被修位置的 `KL(T||S)`，但在高 s2 状态上 downstream s2 无实质变化；bridge 同时改善 downstream s2 和 verifier。
- H3：相同 teacher leg 只有进入后续 context 才产生完整收益；normal bridge 优于 detached。
- H4：状态—动作对应本身有价值；RvI 优于动作数、位置、长度和成本均相同的 A2 shuffled。

端到端 E1/E2 是外部确认，不替代 H1–H4 的机制识别。

## 2. 执行门与停止规则

按以下顺序执行：

1. 数据、许可证、模型 tokenizer 和泄漏审计。
2. D1 信号定标，冻结全部阈值和 token artifacts。
3. D2 paired continuation。W1 不通过则停止昂贵训练，转为 negative/workshop 结果。
4. D0、D3、A2。D0 interaction 不成立则不主张 state-dependent routing；D3 不成立则不主张 context change 是原因。
5. D4/D5 仅作边界扩展，不反向调主路由。
6. 仅在上述选择冻结后运行 E1；E1 通过后再运行 E2。

## 3. D1：信号定标

### 数据与拆分

- 从训练池按 normalized prompt/near-duplicate cluster 做 group split。
- calibration、diagnostic、training 三部分互斥；最终 benchmark 不参与。
- 数学域至少保留 300 个 prompt；若关键 strata 少于 10%，扩充 calibration，而不是降低门槛。
- 医疗域重新定标，严禁复用数学域数值阈值，也严禁接触 HealthBench。

### 计算

- `K_signal=16`，师生 top-K 并集上分别重整化，计算 `KL(T||S)`。
- TA-OPD 式 batch q05/q95 normalization 得到的 `D̃,C̃,D^L,D^I` 只作为当前 batch 的 state descriptor 和兼容性诊断，不能把某个 batch 的数值阈值搬到另一个 batch。
- 主 router 使用 D1 calibration 全集一次性冻结的 raw `D/C` global q05/q95 anchors；以后每个 state 都用同一变换，绝不依赖 inference batch composition。已知的 `token_index/max_response_tokens` 四分位仅做 stability diagnostic 和统计协变量，不产生四套主路由阈值。
- 定义 `s1=max(D^L,D^I)`；它是 disagreement eligibility score，不把它字面解释成“可吸收性”。
- s2 使用未温度缩放的 teacher softmax，在 tokenizer-specific onset token ID 集上求质量。
- high/low 使用 D1 的 global q75/q25；严禁使用 realized/future response length 分层。阈值只可在 D1 修改一次。

### 诊断与通过门

- `|Spearman(s1,s2)| < 0.70`。
- 控制 token position、base difficulty、原始正确性和 `D^I` 后，s2 仍正向预测 bridge benefit。
- `D^L` 正向预测 fixed-context repair gain。
- `D^L-high/s2-low`、`D^I-high/s2-low`、`D^I-high/s2-high` 各占候选状态至少 10%。

保存不可变 `ThresholdArtifact`：split SHA、model revisions、tokenizer SHA、词表 SHA、TRD onset IDs、Relay single-token IDs、global raw `D/C` q05/q95、global q25/q75、position-stratified stability report 和代码 git SHA。任一 fingerprint 不一致即拒绝复用。

## 4. D0：核心 2×2×2 解离

### 因子

- signal type：`D^L-top` vs `D^I-top`。
- state damage：s2-low vs s2-high。
- action：repair vs intervene。

原文的 2×2 不足以识别双信号路由，因此实际预注册为 2×2×2，论文主图可聚焦两个关键对角 strata，但原始八格必须全部公开。

### 状态匹配与随机化

- 至少 1,024 个 states、300 个 prompts；每格至少 128 states/50 distinct prompts。只允许按盲化 outcome variance 与 prompt ICC simulation 上调样本量，并在 action outcomes 解盲前冻结。
- 信号格严格为：`DL-top: DL≥global q75 且 DL>DI`；`DI-top: DI≥global q75 且 DI>DL`；s2-low `≤global q25`；s2-high `≥global q75`。ties 与 middle bands 在随机化前排除并公开数量。
- **冻结-prefix forced-action ITT probe**：在 problem/base difficulty/已知 relative token position/raw divergence 内成组；同一 frozen state 的独立副本只接受一个随机动作，关闭 gate，不与其他 states 共用临时参数更新。统计单位是 problem，不是 token/state。
- **Policy training check**：`repair-only/intervene-only/RvI/A2` 是相互独立的完整训练 run，三个 seeds `{13,17,23}` 只作固定的 seed-level contrasts；不把 3 个 seeds 拟合成随机效应，也不把训练期间 states 当独立重复。

### 动作

- Repair：在原 context 上使用与 D2 相同的 fresh temporary LoRA，对 teacher-top-128 重整化 `q_T` 做 FCE；student 的 `log p_S` 仍来自 full softmax。没有 micro-update 的“loss probe”不能作为即时行为处置。
- Relay eligibility 严格定义为 `φ(c)=1[argmax p_T∈R]·1[TopK_S(c;K=5)∩R=∅]`，`R` 只含 tokenizer 编码后恰为单 token 的 13 个 reflection bases 变体。teacher leg 的首 token 是满足 φ 的 teacher global argmax，而不是“s2 高”或任意 reflection-preferred token。
- Forced-action probe 只在首个 φ-positive trigger 做一条 leg（`M=1`）并强制 student resume，以便得到 post-action outcome。Policy training 使用 `(M,L)=(2,3)`、每 leg 256-token cap、cooldown=1；第 M 条 leg 后 terminal 是 RvI 的已冻结策略选择。独立 Relay baseline 必须保留 upstream 的 K=5/M=2/L=3/cap=256/**无 cooldown**/第 M 条后 terminal 合同。
- Repair 的 `teacher_scored_tokens` quota 按 intervention 每个实际 autoregressive decode position 配平；对 repair 它表示 teacher distribution-scoring positions。这个量是 target/query-position equivalent，不是 FLOP、forward-call 或 wall-time 等价。另做 repair+discarded-sham-generation compute control；所有臂继续报告 prefill、decode、gate 与实测 teacher/student GPU-seconds。
- D0 为保持随机 action 因子可解释，关闭 acceptance gate；gate 只在 A4 和端到端 RvI 中启用。否则被随机到 intervene 的 state 会事后变成 repair，破坏 treatment assignment。

### 主要终点

- problem-level verifier pass / benchmark accuracy。
- H-token continuation s2 residual AUC。
- 被修位置 fixed-context `KL(T||S)` reduction。
- teacher GPU-second、teacher scored token 与 wall-clock Pareto。

### 关键预测

- `D^I-high/s2-high`：intervene > repair。
- `D^I-high/s2-low`：repair 仍产生实质 fixed-context gain，支撑“off-support 不等于不可学”。
- `D^L-high/s2-low`：intervene 不应有更好的成本效益。
- 三重 contrast `Δ3=[(μI,DI,high−μR,DI,high)−(μI,DI,low−μR,DI,low)]−[(μI,DL,high−μR,DL,high)−(μI,DL,low−μR,DL,low)]` 的 95% CI 下界大于 0。

## 5. D2：paired continuation 与 W1

“repair 后由 frozen student 续写”在定义上不完整：若参数和 context 都不变，repair 与 base 必然相同。主协议采用以下三臂：

- Base：`θ0`，原始 prefix。
- Repair probe：每个 state 建 fresh zero-effect LoRA（rank 8、alpha 16、dropout 0；target `q/k/v/o/gate/up/down_proj`），用 AdamW（LR `1e-4`、β=`0.9/0.95`、ε=`1e-8`、weight decay 0、clip 1、FP32 optimizer state）在相同 prefix 做 teacher-top-128 FCE。至少 1 步；marked-position KL 相对下降达到 20% 或最多 8 步后停止，未达到者仍保留在 ITT。随后冻结副本，从原 prefix 续写。
- Bridge：参数仍为 `θ0`，teacher leg 进入 context 后续写。

至少 1,000 states / 300 prompts；每 state 4 个 paired continuation seeds。H 在 D1 的 `{32,64,128}` 中选最小稳定值，之后冻结。H-window（带 EOS mask）只用于 s2 residual；verifier 必须收到独立续写到 EOS 或该领域 max-response 的完整 completion，绝不评分 H-token 残片。

### W1

- Repair 在 marked position 的 KL 下降是与其训练目标同源的 manipulation check，不作为独立发现；仍按 ITT 报告所有 max-step failure。
- Repair 的 downstream s2 change 通过 TOST 落在“无实质变化”区间；不能用 `p>0.05` 当无效证据。
- Bridge 相对 repair 显著降低 s2 residual，并提高 verifier pass。

另用 repair-only 训练 checkpoint 在独立 prefix bank 复测，作为 micro-update probe 的外部有效性检查。4 rollouts/state 与多 state/prompt 使简单 McNemar 不适用；binary 与连续终点都采用 prompt-cluster paired bootstrap（可补充带 prompt cluster 的 GEE）。

## 6. D3：detached 因果对照

只取每条 trajectory 的首个 eligible `φ=1` trigger，固定 `M=1` 并在两臂都强制 student resume。teacher leg 只生成一次，两个 bridge 臂复用 bit-identical token 序列与 loss mask：

- Normal：保留 bridge KV/context，student 从 bridge 之后续写。
- Detached：先从原 prefix 对同一 leg 计算 Relay 的 k1 RKL（target 是**实际发出的 relay token**），随后删除全部 bridge KV、重置 position IDs，再从原 prefix 开一个独立 continuation pass。

D3 同样关闭 gate，避免 context 对照被 action 回滚污染。

审计必须证明 detached 的 post-leg context hash 与 original 一致；把 KV 留在图中但 `stop_gradient` 不算 detached。normal 与 detached 的 teacher token、actual-token k1 RKL、loss mask 和 student continuation length 完全一致。normal 不显著优于 detached 时，撤销“收益来自状态改变”的表述。

## 7. D4：退化盲区

在与其他机制/最终集 group-disjoint 的 s1/s2 双低中建立 repetition challenge bank（至少 500 states/200 prompts；两名盲化标注者确认 loop，κ≥0.70）。旁路阈值只在训练派生 pilot 上冻结，包括滑窗 4-gram 重复率、unique-token ratio、相对 base 的长度膨胀。

“teacher 有替代方向”不是主观标签：从 original prefix greedy 生成 64-token probe，要求它不继续被检测的重复 n-gram，且 4-gram 重复率低于预冻结 escape threshold；flag 在动作/outcome/verifier 之前计算。只有 flag=true 才 intervene，否则 discard。

比较 core router、router+bypass、random budget-matched intervene、discard。主要报告 verified failure recall、loop escape、false trigger、额外成本和 accuracy。该实验失败不杀核心，但旁路不得进入最终方法。

## 8. D5：PACED `p̂=0` rescue

先用 8 个 pure-student rollouts 选出 0/8 prompts，再用独立 32 个 rollouts要求 0/32 确认，并至少得到 200 个同时具有 frozen φ-positive candidate 的 prompt clusters。只允许依据盲化 rescue variance/ICC 在 arm labels 解盲前上调样本量。每臂随后执行一个预注册 action slot；再用第三组独立 `K=8` pure-student rollouts重估 pass rate，最后用第四组 32 rollouts评估，四组 seeds 互斥。

原 PACED 用 `p̂=k/K,w=p̂(1−p̂)`；Jeffreys 臂用 `p̃=(k+0.5)/(K+1),w=p̃(1−p̃)`，`K=8`。比较原 PACED（0 权重）、Jeffreys-smoothed PACED、repair、random bridge、RvI→重估→PACED。所有臂保持相同 batches、optimizer steps 与 student-token budget；原 PACED 记录 zero-gradient sham step。teacher compute 不伪称匹配，按 scored/generated token 与 GPU-seconds 做 Pareto。

在 assignment 前用 base checkpoint 为每个 confirmed prompt 冻结首个 Relay-φ-positive state manifest。Repair 在该位置做 fixed-context top128 FCE；random bridge 在 difficulty/position blocks 内整体打乱预生成 teacher-leg payload/realized-length/cost bundle；RvI 使用冻结 router 与 joint-max gate，gate rejection 仍按 requested-action ITT 保留。这样 action 位置不会被各臂结果事后选择。

主要终点为 rescue@32、`p̂` 变正比例、后续 PACED 权重、相邻 held-out problem transfer。neighbors 必须在看任何 rollout outcome 前按 normalized prompt/equation signature/frozen embedding 预聚类并来自 held-out prompt clusters。仅原训练题变正而无 transfer 时，只能称“解锁训练状态”，不能称能力提升。

## 9. E1：数学主表

### 主设置

- Teacher：`Qwen/Qwen3-4B-Instruct-2507`。
- Student：`Qwen/Qwen3-1.7B`，固定 non-thinking chat template 和 Relay system prompt。
- Train：`BytedTsinghua-SIA/DAPO-Math-17k@65877096c24ffa7abc4e4fa5edb95cf3413a5674` 的英文题。当前 snapshot 有 1,791,700 个物理行，名称中的“17K”不能当样本量。按 normalized/near-duplicate cluster 和 seed `20260828` 做 80/10/10 train/D1-calibration/diagnostic split；冻结并发布 raw/English/unique-cluster/post-denylist/train/calibration/diagnostic counts、cluster membership 与 final split manifest hash。Relay 未公开其精确 English-subset manifest，无法做到 ID 级同集时明示偏差。
- verl、prompt 2,048、response 16,384、T=1、top-p=1、n=1、global batch 128、LR `1e-6` constant、1 epoch。

### 基线与 seed

- 核心 5 seeds `{13,17,23,29,31}`：Vanilla OPD、TA-OPD、Relay、RvI、A2-shuffled。Base 与 Teacher 是 immutable evaluation-only checkpoint，没有“训练 seed”。
- 机制 3 seeds：repair-only、intervene-only、D3 detached、canonical full-vocab TRD、Relay top-128 TRD reproduction。两个 TRD 不能共用一个模糊行名。
- 次级至少 3 seeds：SFT、KD、FastOPD、SKD、TIP、GRPO。
- Teacher 只作 upper bound。

每个 baseline 的可执行合同在 config 冻结：Vanilla 是 student-rollout sampled RKL；TA 是 top-5% `D^L` selector + standard OPD loss；Relay 严格为 K5/M2/L3/cap256/no cooldown/第 M 条 terminal/actual relay-token k1 RKL；canonical TRD 用 full-vocab FKL，Relay reproduction 的 top128 FKL 单独命名。SFT/KD 每 prompt 仅一条 frozen teacher trajectory，KD 若用 top128 必须命名为 adaptation；FastOPD 只在 diagnostic split 从 `{1024,2048,4096,8192}` 选一个 prefix length；SKD 固定 K=5/top128 FKL。TIP 的 selector/loss/hyperparameters 必须在最终评估前从 pinned paper/upstream 完全冻结，否则改名 `TIP-clean-room-adaptation`。

### 评估

- AIME24/25/26、AMC23、HMMT-Feb26、HMMT-Nov25：每题 32 samples。
- MATH500、OlympiadBench：每题 4 samples。
- T=1、top-p=1、max response 32,768。
- `max_model_length=34,817`；冻结 problem IDs 与 sampling seed manifest，并先跨 benchmark 做 normalized/near-duplicate 去重。
- 所有 k-sample 指标叫 `avg@32/avg@4`，定义为 k 个独立 correct indicators 的算术平均，绝不写成 `pass@k`。
- MMLU 固定 2,000 题、5-shot；遗忘门为 `Base−RvI` pp 的单侧 95% CI 上界不超过 1.0 pp，而不是只看点估计。

主终点是八 benchmark macro mean；另报 per-problem pooled accuracy 和成本 Pareto。Relay 是预注册 primary comparator；若要声称“优于所有预注册复现基线”，必须使用 max-stat simultaneous 95% CIs 且 RvI 对每个 non-oracle baseline 的下界均 >0。观察数据后再选“最强基线”的普通 pairwise CI 不支持该表述，更不据此泛称 SOTA。

## 10. E2：医疗主表

- Teacher：锁定 revision 的 Qwen3-32B；Student：Qwen3-4B。先验证相同 tokenizer/vocab；8B 只做成功后的 robustness。
- Confirmatory train 只用 `FreedomIntelligence/medical-o1-reasoning-SFT@fc2c9e8a37b38f38da6d449564a8c350b244aef4` 的 English `medical_o1_sft.json`（file SHA256 `6a0289…05667e`）。字段固定为 user=`Question`，assistant=`Complex_CoT + "\n\n" + Response`，缺字段/非字符串即失败。按 normalized/near-duplicate cluster 以固定 seed 做 80/10/10 train/D1-calibration/diagnostic split并冻结 manifest。
- `avaliev/chat_doctor@19646f30de72c3890c6e0bc67579cbb538076822` 只在主实验成功后做 robustness，仅取 `train.json`，user=`instruction + "\n\n" + input`、assistant=`output`；合并后重新做同一 cluster split，再固定 50:50 source-balanced sampler 与相同 optimizer steps。其 upstream test/validation 不进入训练。两者 HF card 均标 Apache-2.0，但运行前仍要追溯原始来源/条款；card metadata 不自动覆盖上游数据权利。
- HealthBench prompt、rubric、example、reference、grader rationale 全部 denylist；Full 与 Hard 分开报告，Hard 是 Full 子集。
- 主臂：Base、SFT、Vanilla、TA、Relay、repair-only、intervene-only、RvI、A2；TRD 是 d=∞ reference。
- 同名基线沿用 E1 合同，仅替换为 E2 的 data/template/max response：TRD reference 必须是 full-vocab FKL；Relay 必须是 φ/K5/M2/L3/cap256/no cooldown/第 M 条 terminal/actual-token k1 RKL；A2 必须整体打乱预生成的 action/payload/realized-length/cost bundle。各臂不能只留方法名。
- Teacher/student/revision 与 non-thinking chat template 全部冻结。训练臂用 3 seeds `{13,17,23}`；Base 没有训练 seed。每 checkpoint/prompt 生成 1 条 completion，T=1、top-p=1、max response 4,096，同一 sampling manifest 跨臂复用。
- 训练统一使用 pinned verl、prompt 2,048、response 8,192、model length 40,960、T=1、top-p=1、n=1、effective global batch 128、LR `1e-6` constant、1 epoch；C0 只可改变 microbatch/gradient accumulation，不能改变 effective batch。HealthBench 评估 input cap 36,864 + response cap 4,096；任何 overflow 在 preflight 失败并报告，禁止静默截断 benchmark。

官方总分完全使用 `openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47` 的 physician path：grader `gpt-4.1-2025-04-14`、grader max tokens 2,048、主分析 `n_repeats=1`、不改 sampler/template。Full 5,000 prompts 的 completions 直接索引出 Hard 1,000 子集，不另生成一次。Full official score 与预注册 RvI-vs-repair rubric DiD 是 confirmatory；Hard 是重叠的 key secondary。负分 rubric 另报以 `abs(weight)` 加权的 violation rate；不要把正分分母为 0 的子集硬套官方 normalized score。辅助维度、W0 和统计见 [医疗协议](HEALTHBENCH_PROTOCOL.zh-CN.md)。该实验只支持 external-domain behavior，不支持 clinical validity 或 deployment safety。

## 11. Ablations

- A1：full / s1-only / s2-only / no-signal。
- A2：先在看 training/task outcome 前冻结 RvI 的 requested 与 post-gate effective action 及 `(realized bridge length, payload hash, cost signature)`，再在预注册 composite blocks 内整体打乱；destination 不重跑 gate。只打乱 action label，或打乱后让 gate 再改变 realized counts，都不能声称保持实际长度/成本。
- A3：repair-only / intervene-only / discard-free / post-hoc oracle upper bound。
- A4：gate / accept-all / reject-all / s2-only / agree@K-only。
- A5：q70/q75/q80；frozen global vs known-prefix-position-stratified（探索性）。禁止 future/realized response-length strata。
- A6：teacher-top-128、union-top-16、full-vocab FKL、sampled/full RKL。仅当 teacher `q` 在同一 support 重整化且 student probability 仍来自 full softmax 时，FCE 与 FKL 的 student gradient 等价，不重复成臂。
- A7：`M={1,2,3}`、`L={0,1,3,5}`、cooldown、M-th leg 后 terminate/resume。
- A8：reflection lexicon、leave-one-family-out、`K_signal={8,16,32}`。
- A9：no-discard、TRD d=∞、random bridge。
- A10：实际 teacher cost 的三点以上 Pareto sweep。

计算受限时优先 A1、A2、A4、A6。

## 12. 统计方案

- D0 frozen-prefix ITT：匹配块随机化推断，并以 problem-cluster bootstrap/GEE 估计明确编码的 `action×signal_type×s2_band` contrast；position/base difficulty 是预处理协变量。三个 policy-training seeds 只作 fixed seed-level contrasts，不拟合不稳定的 `(1|seed)`，也不把 states 当独立样本。
- D2：连续 s2 与 binary verifier 都用 prompt-cluster paired bootstrap（补充 cluster-GEE）；repair equivalence 用 TOST。简单 McNemar 不适用于每 state 4 rollouts 且每 prompt 多 states 的层级数据。
- E1：两层 bootstrap，先 train seed、再 problem，rollout 嵌套于 problem。
- E2：先重采样 3 个 training seeds、再做 prompt-cluster paired bootstrap，并报 leave-one-seed-out；rubric item mixed model 控制 rubric weight/sign。Base 在各 sampling manifests 下重复评估，但不伪造 training seed。
- D5：每 prompt 的 binomial Clopper–Pearson CI 仅作描述；臂间 rescue、weight 与 transfer 差异用 prompt-cluster paired bootstrap。
- Confirmatory family 用 Holm；探索性 ablation 用 BH-FDR 0.05；bootstrap 10,000 次。

预注册默认实质界：s2 为 high-s2 基线中位数的 10%；数学 macro 1.0 pp；HealthBench normalized score 0.01；MMLU 1.0 pp。可用盲化 variance pilot 调整一次，调整后冻结并记录理由。

## 13. 结论降级矩阵

| 失败点 | 允许的结论 |
|---|---|
| D1 双轴共线或 s2 不预测 bridge benefit | 双信号故事不成立 |
| D2/W1 无解离 | 停止主训练，报告 negative mechanism result |
| D0 interaction 不成立 | 不主张 state-dependent action selection |
| A2 不显著 | 只能称组件组合，不能称路由创新 |
| D3 不显著 | 不主张 context/state change 是收益来源 |
| E1 只胜 Vanilla | 机制方法，不称 SOTA |
| E2 W0 失败 | 官方总分仍有效，rubric 二维仅探索性 |
| D4/D5 失败 | 作为边界条件，不进入方法卖点 |
| target/scored-position、supervised-token 或 optimizer-step 误差 >1% | 撤销 position-count-matched 表述；任何情况下都不称 strict compute-matched |
