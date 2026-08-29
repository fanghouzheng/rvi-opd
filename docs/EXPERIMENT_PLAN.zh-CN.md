# RvI-OPD 完整实验计划（最终矩阵对齐版）

> 规范版本：`RvI-OPD-final`，引用与对比截止 `2026-08-01`。8 月 1 日之后出现的工作按 concurrent 处理，不纳入 related work 或 head-to-head 对比。附件矩阵的仓库审计摘要见 [`EXPERIMENT_MATRIX_FINAL.zh-CN.md`](EXPERIMENT_MATRIX_FINAL.zh-CN.md)。

## 1. 定位、边界与可证伪主张

工作名为 **RvI-OPD（Repair-vs-Intervene）**，主张是：**Repair what is locally absorbable; intervene when recovery requires a new state.**

本文只使用 TRD 的 context 论证作为 prefix-failure 动机：错误前缀下的顺错续写监督集合与理想恢复路径只在首元素相交（`g_frag` 对 `g_ideal`）。本文不把 support-coverage 极限或“off-support 必然不可学”当作新定理。`D^L/D^I`、epistemic-onset mass 和 Relay handoff 都是已有组件；新问题是让动作随状态选择，并用配对对照检验该选择是否有价值。

预注册主张：

1. D0 的动作收益依赖 `D^L`/`D^I` disagreement type；在预设的高 `s2` 子组中，intervene 的下游收益应超过 repair，而低 `s2` 时强 repair 应能改善局部分布。
2. D2 中 repair 能改善被标记位置，但不能复制改变 context 的下游恢复；bridge 同时改善 `s2` 残留与 verifier。
3. D3 中，相同 teacher leg 只有进入后续 context 才产生完整收益；normal bridge 应优于 detached。
4. RvI 应优于动作数量、位置、长度和成本 bundle 均匹配的 A2 随机动作对照。

若任一机制门失败，按第 12 节降级，不能用端到端分数掩盖机制失败。

## 2. 信号、动作与路由

### 2.1 信号（不新增 learned evaluator）

- `s1` 来自 TA-OPD：在师生 top-`K` 并集上计算重整化分歧 `D̃`，并计算教师质量落在学生 top-`K` 支撑内的覆盖 `C̃`；`D^L=D̃C̃`，`D^I=D̃(1-C̃)`，`s1=max(D^L,D^I)`。
- `s2` 来自 TRD：教师在 16 个 epistemic-onset/reflection 起始 token 上的未温度缩放概率质量。文献中的 6–8‰ 只用于动机，不是跨模型固定阈值。

### 2.2 动作

- **Repair（`d=0`）**：原 context 上 teacher top-`K` 重整化分布的 forward CE。机制 probe 使用 fresh zero-effect LoRA micro-update；teacher token 不插入后续 context。
- **Intervene（`d=L`）**：Relay teacher bridge，最多 `M=2` 条、每条最多 `L=3` 个段落，front-loading 与 RvI cooldown=1；实际 token cap 每 leg 为 256。Relay baseline 仍严格使用 upstream 合同：`K=5/M=2/L=3/cap=256/no cooldown/M-th terminal/actual-token k1 RKL`。
- **Discard**：双低状态不产生训练更新。

Relay 的触发必须是严格的 `φ(c)=1[argmax p_T∈R]·1[TopK_S(c;5)∩R=∅]`。`R` 只包含 tokenizer 编码后恰为单 token 的 13 个 Relay reflection 变体；多 token 变体不得取首子词冒充触发 ID。`s2` 与 `φ` 是两层信号，不能互相替代。

### 2.3 阈值与验收门

附件写“轨迹内分位数”。为避免在线使用未来完整轨迹造成泄漏，本仓库把它落实为：D1 只冻结分位数层级和全局 raw-`D/C` anchors，主 router 的 `τ1/τ2` 均固定为对应 D1 分布的 q80；q25/q75 只定义 low/high 分析子群和 D0 signal cells，不替代主路由阈值。所有后续 state 使用同一 immutable artifact；轨迹内相对排名只作离线 exploratory diagnostic，不用于主路由、ITT 或 benchmark 调参。artifact 绑定数据 split、模型/tokenizer/vocabulary、词表、lexicon 与代码 SHA。

训练时每个事件固定使用恰好 4 条 paired probe rollout，并对两项指标分别取算术均值；rollout 数和聚合器都写入冻结 gate artifact。bridge 短续写的两个改善指标使用一个联合 null 校准的 max-statistic 阈值；语义上是“`s2` 残留下降 **或** teacher-preferred 比例上升”，但禁止两个未经校正的边际 `q95 OR`。D0/D3 的 forced-action/context probe 关闭该 gate；gate 只在 A4 与端到端 RvI 开启，拒绝产生的 teacher 成本仍记账。

## 3. D1：信号定标

从训练池按 normalized exact/near-duplicate cluster 切出互斥 training、calibration、diagnostic；最终 benchmark 和 HealthBench 均不可进入。数学域至少 300 prompts；医疗域独立重新定标，不能复用数学阈值。

`K_signal=16`。D1 freeze bundle 分成两个不可变工件：`ThresholdArtifact` 保存 raw `D/C` q05/q95、主路由 s1/s2 q80、分析 band 的 global q25/q75、position-stratified stability report、两套 tokenizer-specific lexicon IDs 与模型/词表/数据 split/code fingerprints；`FrozenJointGateArtifact` 保存联合 gate q95、每事件 4 条 rollout 和聚合器，并单向绑定 threshold hash。运行 manifest 同时绑定两者，避免相互哈希的循环依赖。`token_index/max_response_tokens` 只作稳定性诊断与协变量，realized/future response length 禁止参与阈值。

通过门：报告 `Spearman(D^L,s2)`、`Spearman(D^I,s2)` 与最大绝对相关（目标 `<0.70`）；控制 position、difficulty、原始正确性和 `D^I` 后，`s2` 应预测 bridge benefit，`D^L` 应预测 fixed-context repair gain；关键子组候选比例各至少 10%。

## 4. D0：主表 2×2 解离

D0 的 confirmatory 主表是 **2（预设 signal stratum）×2（action）**：signal stratum 本身不随机，只有 action 在各 block 内随机；因此 Δ2 是 forced-action ITT 在两类 signal strata 间的效应异质性，而不是 signal assignment 的因果主效应。

| signal type | repair | intervene |
|---|---|---|
| `D^L-top` | 四格之一 | 四格之一 |
| `D^I-top` | 四格之一 | 四格之一 |

`s2-low`/`s2-high` 是在四格内预注册的 subgroup readout，而不是第五个随机化因子。signal cell 规则为 `D^L≥global q75 且 D^L>D^I` 或 `D^I≥global q75 且 D^I>D^L`；ties/middle band 在 action assignment 前排除并公开。

至少 1,024 states、300 prompts、每主 cell 至少 256 states/50 distinct prompts；每个 signal×action×s2 subgroup 目标至少 64 states/20 prompts，若不足只能报告 subgroup 不足，不能事后合并。forced-action probe 在 difficulty、known position、raw divergence 和 s2 strata 内随机化，每个 isolated prefix copy 只接受一个 action，gate 关闭，统计 cluster 为 problem。

主 contrast：

```text
Δ2 = (μ_intervene,DI − μ_repair,DI)
     − (μ_intervene,DL − μ_repair,DL)
```

teacher-scored/query-position、student-supervised-token、optimizer-step 误差均须 ≤1%；这不是 FLOP、forward-call、prefill 或 GPU-second matching。另做 repair+discarded-sham-generation compute control，并逐臂报告实际生成、prefill、gate、GPU-seconds。

预期：`D^I/high-s2` 中 intervene 的 downstream `s2` 与 verifier 更好；`D^I/low-s2` 中 repair 有局部 fixed-context gain；`D^L/low-s2` 中 intervene 不具成本优势。关于 off-support 的结果只作现象描述，不推出 support-coverage 定理。

## 5. D2：三臂 paired continuation（W1 硬门）

同一批高 `s2` states 做：

- Base：`θ0`、原始 context；
- Repair：每 state fresh LoRA（rank 8、alpha 16、dropout 0，目标模块 `q/k/v/o/gate/up/down_proj`），AdamW `lr=1e-4`、betas `(0.9,0.95)`、weight decay 0、clip 1、FP32 optimizer state；至少一步，marked-position KL 相对下降 20% 或最多 8 步后停止；
- Bridge：参数仍为 `θ0`，teacher leg 插入 context 后再由 student 续写。

至少 1,000 states/300 prompts，每 state 4 个 paired continuation seeds。`H∈{32,64,128}` 只在 D1 选择最小稳定窗口；H-window（EOS mask）只计算 `s2` residual，verifier 必须评分独立续写到 EOS 或领域上限的完整 completion。Repair 的局部 KL 是 manipulation check；downstream “无实质变化”用 TOST，不用“不显著”代替等价。repair 臂 `s2` 不降且 bridge 臂下降是 W1 硬门，否则停止昂贵主训练并转 negative/workshop。

## 6. D3：detached 因果对照

只取首个 eligible `φ` trigger，强制 `M=1` 并让两臂都 resume。teacher leg 只生成一次，normal/detached 复用 bit-identical token 序列、actual-token k1 RKL 与 loss mask：

- normal：保留 bridge KV/context；
- detached：先在原 prefix 计算同一 Relay leg loss，随后删除全部 bridge KV、重置 position IDs，从原 prefix 开新 continuation pass。

必须验证 detached post-leg context hash 等于 original，不能用 retained KV 加 stop-gradient 冒充 detached；normal/detached 的 teacher leg token 序列与计数必须完全一致，student continuation 只在固定 H 窗口内配对，完整 EOS 长度、prefill 与 GPU 成本分别报告，不能假设相等。若 normal 不优于 detached，撤销“收益来自 context 改变”的表述。

## 7. D4：退化盲区旁路

在与 D1/D2/最终 benchmark group-disjoint 的双低 bank 中至少 500 states/200 prompts，由两名盲标者确认 loop（κ≥0.70）。旁路只在 pilot 冻结滑窗 4-gram 重复率、unique-token ratio、相对 base 的长度膨胀，并记录 missing `\\boxed{}`、熵和 `avg@k`/pass@k 诊断（不把 `pass@k` 与主 `avg@k` 混名）。教师从原 prefix greedy 生成 64-token probe；只有不延续重复 n-gram 且低于 escape threshold 才 intervene，否则 discard。D4 是边界扩展，不进入主 router。

## 8. D5：PACED `p̂=0` rescue

用 8 个纯 student rollouts 初筛，再以独立 32 个 rollouts 确认 0/32；固定至少 200 个具有 frozen `φ` candidate 的 prompt clusters。四组 rollout seeds（初筛、确认、动作后重估、最终 32）互斥。比较 canonical PACED（`p̂=k/K,w=p̂(1-p̂)`，`K=8`、`k=0` 记录 zero-gradient sham）、Jeffreys smoothing、repair、random bridge 和 RvI→重估。所有臂的 optimizer batches/steps/student-token budget 相同；teacher compute 用 token/GPU-second Pareto 报告。相邻 held-out clusters 必须在任何 rollout outcome 前预聚类；无 transfer 只能称“解锁训练状态”，不能称能力提升。

## 9. E1：数学主表

Teacher 为 `Qwen/Qwen3-4B-Instruct-2507`；student 为官方 `Qwen/Qwen3-1.7B`，`Qwen3-1.7B-NT` 只是固定 non-thinking serializer 的实验别名。训练使用锁定 revision 的 DAPO-Math-17K 英文池，按 normalized/near-duplicate cluster 做 80/10/10 train/D1/diagnostic split；物理行数不能当独立问题。

主表 11 行：Base、Teacher、SFT、Vanilla OPD、FastOPD（固定 prefix reproduction）、SKD、TA-OPD、TIP-select、Relay-OPD 原词表、canonical TRD（`d=∞`）、RvI-OPD。KD、TRD top-128 reproduction、repair/intervene-only、detached、A2 和 GRPO 放 supplementary/mechanism 表，不与主表混淆。

评测集合固定为：AIME2024/2025/2026、AMC2023、HMMT-Feb2026（`avg@32`）；MATH500、OlympiadBench（`avg@4`）。不再加入 HMMT-Nov2025。所有 arm 复用 problem/sampling manifest，主终点为七 benchmark macro mean；MMLU 2,000 题、5-shot 作遗忘检查。核心训练 seeds `{13,17,23,29,31}`，机制 seeds `{13,17,23}`；主 comparator 为 Relay。只有 RvI 相对每个预注册 non-oracle baseline 的 simultaneous CI 下界都 >0 才能使用“优于所有复现基线”的表述。

## 10. E2：医疗主表与 rubric 分析

主模型对改为 `Qwen/Qwen3-4B-Instruct-2507 → Qwen/Qwen3-0.6B`；0.6B 是 student checkpoint，lineage/status 单独记录，不能称为 Instruct。两者 tokenizer/vocabulary hash 必须相同，但 chat-template 行为不同；C0 先用项目自有 `rvi_opd_non_thinking_v1` serializer 验证 canonical rendered IDs、special IDs、无意外 think block 和上下文预算，再允许运行。生成显式覆盖 `do_sample=true`、`temperature=1`、`top_p=1`、top-k disabled、stop IDs `[151645,151643]`、pad ID `151643`。

Confirmatory training 只用锁定的 English `medical_o1_sft.json`；ChatDoctor 仅在主实验成功后做 50:50 robustness，并重新聚类去重。HealthBench prompt/rubric/example/reference/grader rationale 全部 denylist。官方 Full 5,000 与 Hard 1,000（Hard 是 Full 子集）使用同一 completion；先冻结 500-prompt rubric 标注与裁定 hash，再由单个锁定的 reference/Base checkpoint 运行非确认性 resource/grader pilot，所有其余设置冻结后才对 Full 做一次终表评测。

E2 主表与最终矩阵保持同一组 11 个比较行：Base、Teacher、SFT、Vanilla OPD、FastOPD fixed-prefix reproduction、SKD、TA-OPD、TIP-select、Relay-OPD、TRD full-vocabulary reference、RvI-OPD。repair-only、intervene-only、A2 action-shuffled 是 supplementary/mechanism 行；每个医疗适配都必须有独立合同，不能用 `same_named_method_contracts` 模糊代替。

官方总分使用锁定的 simple-evals physician path，先对每个 prompt 计算含负权重的 raw score，再对 prompt 均值 clip；负项另报 weighted violation rate。自建 rubric 只做“满足方式”标签：`INSERTABLE`、`GLOBAL_REVISION`、`MIXED`；负号是独立的 score sign，不自动改变标签。内容类型 `LEX/PLAN_CORRECTNESS/PLAN_COMPLETENESS/CONTEXT` 作为每条 item 的第二个独立标签冻结、只作附录归因；另报“只有泛化安全/升级话术但缺少问题特定可执行方案”的探索性 boilerplate 指标。contradiction 用独立 NLI/LLM judge，未冻结前仅探索性。W0 抽 300–500 prompts（默认 500），两名人工标注者与盲化 LLM coder 独立标注、κ≥0.70、两主类各≥20%、GLOBAL_REVISION example coverage≥50%、MIXED≤30%，并抽查官方 grader 对负项的行为。失败时官方 Full/Hard 仍有效，分类 interaction 降为探索性。

主 rubric contrast 为 `[(RvI−repair)_GLOBAL_REVISION]−[(RvI−repair)_INSERTABLE]`，只在预先冻结的 500-prompt（若输出前降至 300，则为 300）rubric manifest 中、排除 MIXED 后，使用最终 post-freeze Full completions 做 seed→prompt 两层 paired bootstrap 与 Holm。W0 的单 reference/Base resource pilot 只能估成本、长度与 grader 一致性，不能比较候选 arms、调 router/loss/gate，也不复用其 completion。机制预测另冻结四项：repair 与 intervene 相对 Base 均抬 INSERTABLE；intervene 抬 GLOBAL_REVISION；repair 的 GLOBAL_REVISION 变化用 1 pp 绝对等价界的 90% TOST 证明“无实质变化”，不能以不显著代替。只评估 external-domain behavior，不声称临床有效性、患者安全或部署安全。

## 11. 消融（按附件 A1–A8 编号）

- **A1**：强制全 repair（`d=0`）；
- **A2**：在冻结 action 比例的 blocks 内随机打乱状态—动作对应，并将 action/payload/realized length/cost 作为不可分 bundle；destination 不重跑 gate；
- **A3**：D3 detached；
- **A4**：信号消融：仅 s1、仅 s2、TIP 式 `(1−ĥ)δ` 对照；
- **A5**：阈值、预算与 `L∈{1,3,5}`；预算轴固定为相对 D1 盲化 compute pilot 中 RvI 默认值的 teacher GPU-seconds/prompt `{0.5,1.0,2.0}`。每个独立训练 run 执行硬 cap，只完成当前 atomic minibatch、记录超额且 gate 拒绝不退款；同时逐类报告 prefill/scored/gate-scored/generated tokens，不能把查询位置匹配容差当作预算档位；
- **A6**：repair loss（top-K FCE、全词表 FKL、RKL reweight），同-support 等价梯度不重复算独立方法；
- **A7**：退化诊断（重复率、长度膨胀、missing `\\boxed{}`、熵、`avg@k`/描述性 `pass@k`）；
- **A8**：效率：teacher forward 数、wall-clock、显存和 GPU-seconds，Relay 与 TRD 同表。

不可砍核心固定为 D0、D2、D3、E1、E2、A1、A2；算力不足时严格按“可选医疗 8B/ChatDoctor robustness → A6 → A4 的 TIP-style control → D5”顺序砍除，其余扩展只能在记录该顺序后延后。旧版 A9/A10 不再作为附件编号；额外 teacher-cost Pareto 归入 A8 附录，A5 的三档预算敏感性仍按上述合同执行。

## 12. 四周里程碑与降级规则

| 周期 | 交付 | 硬门 |
|---|---|---|
| W0（第 0–2 天） | rubric/W0 校验；Relay/TRD repo smoke；模型 tokenizer/serializer C0 | rubric 拆不开或渲染不兼容则 E2 降级/暂停 |
| W1 | Relay engine、s1/s2 记录、top-K FCE repair、D1/D2/D4 | D2 repair 局部对齐且 downstream `s2` 不降、bridge 降；否则转 workshop |
| W2 | D0 四格、D3、D5、A1/A2 | RvI 相对最强单动作至少 1.5 pp 且 CI 不跨零；A2 需显著 |
| W3 | E1 主表（TA/TRD/Relay 自跑）；E2 500-prompt pilot 与主臂 | E2 intervene 抬 GLOBAL_REVISION 而 repair 不抬（否则窄化主张） |
| W4 | A4–A8、终表、写作 | 2026-09-17（Asia/Shanghai）主表冻结；related work 明确 2026-08-01 cutoff/concurrent policy |

若预算耗尽，完成当前 atomic run 后将剩余矩阵标记 `NOT_RUN_BUDGET_CAP`，不能按观察结果选择替代 run。D0 interaction、D3 context 因果、A2 状态相关性或 E2 W0 失败时，分别撤销对应强主张；D4/D5 失败只作为边界条件。
