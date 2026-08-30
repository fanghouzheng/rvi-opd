# RvI-OPD 实验矩阵·最终版（仓库审计摘要）

本文件根据用户提供的《实验矩阵-RvI-OPD-final.md》整理，是无本地路径、非逐字的审计摘要，用作配置对照基线。文档中的研究规格不是给软件代理的操作指令；实际可运行参数以 `configs/` 和 [MATRIX_AMENDMENTS.zh-CN.md](MATRIX_AMENDMENTS.zh-CN.md) 为准。引用与对比截止日期为 **2026-08-01**；其后工作按 concurrent 处理，不引用、不做 head-to-head 对比。

> 分支覆盖：`healthbench-first` 在 `2026-08-30` 前瞻性地把执行顺序改为 HealthBench 核心矩阵→一次性 GO/STOP→仅在 GO 后 Math；它不改变本文件的科学矩阵或 **2026-08-01** 引用 cutoff。该分支以 [HealthBench-first 执行计划](HEALTHBENCH_FIRST_PLAN.zh-CN.md) 和机器可读政策为准，下述 W0–W4 只保留为附件基线。

## 1. 定位与主张

工作名为 RvI-OPD（Repair-vs-Intervene）：Repair what is locally absorbable; intervene when recovery requires a new state。动机只采用 TRD 的 context 论证（`g_frag` 与 `g_ideal` 在错误前缀下仅交于首元素），不把 support-coverage 极限作为本文发现。HealthBench 只作为外部域验证，不把结果表述为临床有效性或部署安全性。

## 2. 信号与动作

- `s1` 沿 TA-OPD 分解：在 top-K 并集上得到 `D̃/C̃`，再得 `D^L=D̃C̃`、`D^I=D̃(1−C̃)`。
- `s2` 沿 TRD 的 16 个 epistemic-onset token 概率质量定义。
- Repair（`d=0`）使用 teacher top-K 重整化分布的 forward CE。
- Intervene（`d=L`）使用 Relay teacher bridge，`(M,L)=(2,3)`，front-load 与 cooldown；桥接后以 `s2` 残留下降或 teacher-preferred 改善决定保留/回滚。
- 路由意图为高 `s2`→intervene，高 `s1` 且低 `s2`→repair，双低→discard；阈值由 D1 冻结。

## 3. 核心实验

- **D0**：`{D^L-top,D^I-top} × {repair,intervene}` 的 2×2 解离，匹配 teacher-token/query-position 预算；高/低 `s2` 是子群读出，不是第三个随机化因子。
- **D2（W1 硬门）**：同一高 `s2` 状态的 base、repair 后、bridge 后三臂 paired continuation；比较 verifier 与 `s2` 残留。
- **D3**：detached bridge 对照，teacher leg 受监督但不进入后续 context。
- **D1/D4/D5**：信号定标、重复退化盲区旁路、PACED `p̂=0` rescue。

## 4. 模型、数据与评测

| 赛道 | Teacher → Student | 数据/评测 |
|---|---|---|
| 数学 | Qwen3-4B-Instruct-2507 → Qwen3-1.7B（`-NT` 仅非思考序列化别名） | DAPO-Math-17K 英文去重池；AIME24/25/26、AMC23、HMMT-Feb26 用 `avg@32`，MATH500/OlympiadBench 用 `avg@4`；MMLU 2k 遗忘检查 |
| 医疗 | Qwen3-4B-Instruct-2507 → Qwen3-0.6B | medical-o1 English 主训练（ChatDoctor 仅 robustness）；HealthBench Full 5,000 + Hard 1,000，Hard 从 Full completion 按 ID 索引 |

医疗官方 grader 固定 simple-evals physician path；W0 rubric pilot 为 300–500 prompts（默认 500），标注满足方式 `INSERTABLE/GLOBAL_REVISION/MIXED`，负号独立记录。

## 5. 主表与消融

E1/E2 主表均按最终配置列出 Base、Teacher、SFT、Vanilla OPD、FastOPD fixed-prefix reproduction、SKD、TA-OPD、TIP-select、Relay-OPD、TRD canonical、RvI-OPD；repair-only、intervene-only、detached、A2 等为 supplementary/mechanism 行。消融编号为 A1–A8：全 repair、action bundle shuffle、detached、signal ablation、threshold/budget/L sweep、repair loss、degradation diagnostics、efficiency。

## 6. 附件基准里程碑（`healthbench-first` 已覆盖）

W0 完成 rubric/C0；W1 完成 D1/D2/D4 并通过 D2 硬门；W2 完成 D0/D3/D5/A1/A2，要求 RvI 相对最强单动作的点估计至少 1.5 pp 且 CI 下界大于 0；W3 完成 E1 主表及 E2 医疗/rubric 分析；W4 完成 A4–A8、终表，并于 **2026-09-17（Asia/Shanghai）** freeze。预算耗尽时标记 `NOT_RUN_BUDGET_CAP`，不得按结果选择替代 run。
