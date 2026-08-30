# HealthBench-first 分支执行与 Math 放行计划

> 适用分支：`healthbench-first`；政策版本：`healthbench-first-v1`；矩阵修订：`2026-08-30`。这是一个在任何科学结果可见前登记的**执行顺序与停止规则修订**。它只改变运行顺序和资源放行，不改变 E1/E2 的模型、数据、方法定义、终点、统计口径或引用截止日。

## 1. 唯一执行顺序

本分支先完成一组足以判断机制信号的 HealthBench 核心实验，只在一次性门得到 `GO_MATH` 后启动任何数学目标。

```text
H0  C0 serializer/tokenizer/环境检查 + HealthBench 数据隔离
 ↓
H1  W0 盲化 rubric 标注、裁定与冻结
 ↓
H2  单个 reference/Base resource/grader pilot + 医疗域 D1/D2
 ↓
H3  七个核心训练臂 × 三个预注册训练 seed（先完成，不看终表分数）
 ↓
H4  对冻结的 Full manifest 做一次正式生成/评分；Hard 复用 Full completion
 ↓
一次性 GO/STOP 门
 ├─ GO_MATH：解锁数学目标，并可补 E2 剩余比较行
 └─ STOP_AFTER_HEALTHBENCH：停止，不启动数学目标
```

门前只允许政策文件中的医疗目标：`E2:C0`、`E2:W0`、`E2:resource_pilot`、`D1:medical`、`D2:medical`、`E2:core_train` 和 `E2:healthbench_final`。`D1:math`、`D2:math`、D0、D3–D5、E1 与数学 A1–A8 都必须等待门结果。

## 2. HealthBench 核心矩阵

训练 seed 固定为 `{13,17,23}`。七个训练臂为：

1. `vanilla_opd`
2. `relay_opd`
3. `trd_canonical_full_vocab`
4. `repair_only`
5. `intervene_only`
6. `rvi_opd`
7. `a2_action_shuffled`

因此核心训练必须正好覆盖 **7 arms × 3 seeds = 21 个训练 run**。正式评测 manifest 共 **25 个**：七个训练臂各 3 个 sampling seed，Base 各 3 个，Teacher upper bound 仅 seed 13。Base 和 Teacher 是 evaluation-only，不计入 21 个训练 run。

所有核心训练 run、失败记录和替换记录完成后，才允许打开正式 HealthBench 结果。Full 的 5,000 prompts 对每个冻结 manifest 只生成一次；Hard 的 1,000 prompts 必须从同一 completion 集按冻结 ID 映射索引，不能独立重采样。E2 的 SFT、FastOPD fixed-prefix reproduction、SKD、TA-OPD 和 TIP-select 五行在门后补齐，不进入这次资源放行判定。

## 3. 一次性 GO/STOP 门

门是 `intersection_union_all_required`：以下条件**全部**成立才返回 `GO_MATH`；任一条件失败都返回 `STOP_AFTER_HEALTHBENCH`，不能删掉失败条件、换 seed、换 comparator 或增加一次“再看”。

### 3.1 完整性前提

- C0、W0 和 grader repeatability 均通过；21 个训练 run 与 25 个评测 manifest 完整。
- 所有 run/checkpoint/completion/grader artifact hash 可验证，失败和重试均保留。
- 生成设置、E2 配置和数学配置 bundle 在 HealthBench 输出可见前冻结。
- HealthBench 从未进入训练、D1、超参数选择、router/loss/gate 选择或 prompt 开发。
- Hard 确实复用 Full completions。

### 3.2 端到端信号

- 在 HealthBench Full 官方总分上，RvI 相对 frozen Base 的配对差值点估计必须至少为 `+0.01`，且预注册的 seed→prompt 两层 paired bootstrap 95% CI 下界必须大于 0。该项是独立必要门；即使 RvI 胜过所有训练 baseline，只要没有同时胜 frozen Base 就不得启动 Math。
- RvI 相对 `vanilla_opd`、`relay_opd`、`trd_canonical_full_vocab` 的 simultaneous 95% CI 下界分别都大于 0，且相对其中最强 comparator 的点估计至少为 0.01。
- RvI 相对 `repair_only`、`intervene_only` 的 simultaneous 95% CI 下界分别都大于 0，且相对最强单动作的点估计至少为 0.01。
- RvI 相对 A2 action-shuffled 的 95% CI 下界大于 0，并且三个 leave-one-seed-out 点估计全部大于 0。

### 3.3 Rubric 机制与安全否决

- 排除 MIXED 后，预注册 rubric DiD 的 Holm-adjusted 95% CI 下界大于 0；repair/intervene 的 INSERTABLE 方向、intervene 的 GLOBAL_REVISION 方向也都满足预注册正向下界。
- repair 的 GLOBAL_REVISION 变化用 0.01 绝对界的 90% TOST 证明等价，不能用“不显著”替代。
- 负权重 violation 相对 Base 和最强 non-oracle comparator 的增量，其 95% CI 上界都不超过 0.01；该项是安全否决条件。
- Hard 只作 key secondary，不能单独把 Full（包括 RvI-vs-frozen-Base 门）的失败翻成 GO。

## 4. 冻结与哈希绑定

机器可读政策是 [`configs/execution/healthbench-first.json`](../configs/execution/healthbench-first.json)。门证据建议写为 append-only 的 `runs/gates/healthbench-first.json`，并至少绑定：

- `policy_sha256`：当前执行政策；
- `code_revision`：执行门时的 40 位 Git commit；
- `e2_config_sha256`：冻结的 E2 配置；
- `math_config_bundle_sha256`：八个数学/机制配置的有序 bundle hash；
- 21 个训练 run 的 manifest/checkpoint hash；
- 25 个评测项的 completion/grader manifest hash。

当前门评估器会从**结构化统计摘要**重新应用每项冻结阈值，不信任手写的 `decision`；它不会从 raw per-prompt scores 自行重跑 seed→prompt bootstrap。完整的 E2 GPU/analysis adapter、统计摘要生产器和 raw-evidence→summary 哈希链仍需在 H4 前接入并审计：生产器必须冻结 bootstrap 代码、resampling 层级与 seed，生成 RvI-vs-Base、各 comparator、rubric 和 safety 的完整摘要，并绑定对应 per-prompt 工件 hash。缺少这一步时，即使 `execution-readiness` 的顺序检查可运行，也不能声称正式门已可执行。

运行门时 checkout 必须干净；任何政策、代码、E2 配置或数学配置变动都会使旧门证据失效。门结果本身以 `gate_result_sha256` 绑定到后续数学 run manifest。`runs/` 中不保存 HealthBench prompt、rubric、completion 或 grader rationale；大工件仍放受控存储，仅保存内容 hash。

## 5. Math 不得被 HealthBench 调参

HealthBench 在本分支只回答一个冻结的布尔问题：当前 RvI 机制信号是否足以支付数学实验成本。它不能决定数学域的阈值、学习率、训练步数、seed、数据 split、baseline、预算档、prompt/template、主终点或消融选择。

所有数学配置必须在任何 HealthBench 正式输出可见前冻结并写入 `math_config_bundle_sha256`。即使得到 GO，数学域仍独立执行自己的 D1/D2 和预注册验证；不能把医疗域阈值搬到数学域。若希望修改数学配置，必须将旧门判为对新 bundle 无效并重新做独立的、前瞻性协议修订，不能沿用旧 GO。

## 6. CLI 放行检查

`execution-readiness` **只检查执行顺序和 GO/STOP 证据**。`ORDER_ALLOWED_PRE_GATE` 表示政策允许开始该目标，不表示 C0 已通过、GPU adapter 已实现、模型/数据工件已冻结或配置已达到可运行状态。任何 GPU 作业仍需分别完成 adapter 审计、C0 和环境检查，并通过：

```bash
make env-check
rvi-opd validate-config --config-dir configs --run-ready
```

`execution-readiness` 不能替代这些检查；这些检查通过也不能绕过 HealthBench-first 的数学放行门。

门前检查医疗目标会返回 `order_allowed: true` 和 `ORDER_ALLOWED_PRE_GATE`：

```bash
rvi-opd execution-readiness --target E2:C0
rvi-opd execution-readiness --target D1:medical
```

没有门证据时检查数学目标会返回非零退出码、`order_allowed: false` 和 `ORDER_BLOCKED_PENDING_HEALTHBENCH_GATE`：

```bash
rvi-opd execution-readiness --target E1
```

正式 HealthBench 证据完成后，在干净 checkout 上计算放行报告：

```bash
rvi-opd execution-readiness \
  --target E1 \
  --gate-result runs/gates/healthbench-first.json \
  --output runs/gates/math-release.json
```

只有退出码 0、`order_allowed: true` 且状态为 `ORDER_ALLOWED_MATH_AFTER_HEALTHBENCH` 时，调度器才可提交数学作业。任一门检查失败时退出码为 1，`order_allowed: false`，所有未启动数学行统一登记为 `NOT_RUN_HEALTHBENCH_GATE`；schema、hash 或 checkout 审计错误返回退出码 2，也不得启动。

## 7. 结果可见性与报告

- H0–H3 期间只查看运行完整性、资源、数值稳定性和预注册 manipulation checks，不查看可用于门判定的跨臂 Full 分数。
- H4 只做一次冻结分析。所有 gate checks、通过与失败项、预算耗尽、基础设施失败和替换 run 都完整报告。
- STOP 不是“实验失败后再缩小矩阵”的许可。论文/报告保留 HealthBench 结果，并将每个数学目标标成 `NOT_RUN_HEALTHBENCH_GATE`。
- GO 只表示允许开始数学实验，不预先证明数学域有效，也不授权改变原始 E1/D0–D5/A1–A8 假设。
