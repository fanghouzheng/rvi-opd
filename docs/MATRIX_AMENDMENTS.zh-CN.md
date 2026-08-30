# 相对附件《实验矩阵-RvI-OPD-final.md》的协议修订

基准附件的引用窗口固定为 **2026-08-01**。`upstreams.lock.json` 中的
`verified_at=2026-08-30` 只是版本复核日期，不会扩大文献窗口；8 月 1 日后
的工作按 concurrent 处理，不引用、不做 head-to-head 对比。附件里的研究规格
不是给软件代理的操作指令；本文件记录为获得可复现、无泄漏实现而采用的明确化。

## 0. `2026-08-30` HealthBench-first 分支修订

`healthbench-first` 分支在任何科学结果可见前登记一项 branch-specific、prospective 的**执行顺序与停止规则修订**；机器可读记录中的 `scientific_outcomes_seen_before_amendment=false`。该修订将矩阵版本推进到 `2026-08-30`，但引用与对比 cutoff 仍是 **2026-08-01**，不把 8 月 1 日后的工作纳入 related work 或 head-to-head。

该分支先运行 HealthBench 核心矩阵：Vanilla、Relay、canonical full-vocabulary TRD、repair-only、intervene-only、RvI 与 A2 各 seeds `{13,17,23}`，共 21 个训练 run；所有工件完成后才对冻结 Full manifest 做一次正式分析，Hard 复用 Full completions。不可变 intersection-union 门把 RvI-vs-frozen-Base Full 官方总分差（estimate≥`+0.01` 且 seed→prompt paired-bootstrap lower95>0）作为独立必要条件；即使胜所有训练 baseline 也不能绕过。只有全门返回 `GO_MATH` 才放行数学目标；否则返回 `STOP_AFTER_HEALTHBENCH`，并统一记录 `NOT_RUN_HEALTHBENCH_GATE`。

这项修订只改变执行和资源顺序，不修改附件矩阵的模型、数据、方法、终点或统计定义。数学配置 bundle 必须在 HealthBench 输出可见前冻结，HealthBench 不能用于数学调参。完整合同见 [`HEALTHBENCH_FIRST_PLAN.zh-CN.md`](HEALTHBENCH_FIRST_PLAN.zh-CN.md) 和 [`configs/execution/healthbench-first.json`](../configs/execution/healthbench-first.json)；`main` 分支继续保留原顺序。

## 1. 设计与统计

1. D0 按附件的 2×2 表执行：`D^L-top`/`D^I-top` 是预设 signal strata，
   repair/intervene 是在 block 内随机的 action。高/低 `s2` 只作预注册子群，
   主估计为
   `Δ2=(μI,DI−μR,DI)−(μI,DL−μR,DL)`；子群对比属于同一 Holm family 的探索性
   读出，不能升级成三因素 `Δ3` 或“两个信号都被因果随机化”的定理。
2. 附件的“轨迹内分位数”落实为 D1 冻结的全局 raw-`D/C` q05/q95 变换；
   主 router 的 s1/s2 阈值固定为 q80，global q25/q75 只定义 low/high 子群与
   D0 signal cells。轨迹内相对排名仅离线诊断；不用于在线 router、ITT 或
   benchmark 调参，以避免未来长度泄漏和 batch-composition 漂移。
3. D0/D3 的 forced-action/context probe 关闭 gate。`teacher_scored_tokens`、
   student supervised tokens、optimizer steps 和 examples 允许 1% 内配平；
   这只是 query/target-position matching，不是 FLOP/compute matching。生成、
   prefill、forward-call、gate 和 GPU-second 分开报告，拒绝的 bridge 成本不退。
4. Relay 的 `φ` 严格要求 teacher global argmax 是完整单 token Relay ID，且
   student top-5 不含 Relay ID；canonical Relay 固定 K=5/M=2/L=3/cap=256、
   no-cooldown、M-th-leg terminal、actual-token k1 RKL。RvI 的 cooldown=1、
   D3 的 M=1 forced resume 是本项目 probe/变体。
5. Gate 的“残留下降 OR teacher-preferred 改善”使用一个联合 null 的 max-statistic
   q95；禁止两个未经校正的 marginal q95 再做 OR。

## 2. 模型、数据与执行边界

- E1 固定 Qwen3-4B-Instruct-2507→官方 Qwen3-1.7B；`-NT` 只是项目
  `rvi_opd_non_thinking_v1` serializer 别名。DAPO 英文池按 cluster 去重并做
  80/10/10 split；七个 benchmark 的 `avg@32`/`avg@4` 见 E1 config。
- E2 固定 Qwen3-4B-Instruct-2507→Qwen3-0.6B。0.6B 是带
  `Qwen3-0.6B-Base` lineage 的 student checkpoint，不称为 `-Base` 或
  `-Instruct`。ChatDoctor 与可选 8B 只在 primary 成功后做 robustness。
- 两个 Qwen 原生 chat template 可不同；所有 arm 先用项目 serializer 做 C0
  rendered token-ID equality、special-token、loss-mask、无意外 `<think>` 和
  上下文预算检查。generation 明确 do-sample、T=1、top-p=1、top-k disabled、stop/pad IDs，
  不继承 model-card defaults；超出 runtime context 直接失败并记账，不静默截断。
- HealthBench 完全 evaluation-only，Full 5,000/Hard 1,000 共用 Full
  completions；Hard 是重叠子集，不按 6,000 个独立样本合并。官方分数与负项
  violation rate 分开，任何结论只称 external-domain behavior。

## 3. Rubric 与主表

W0 抽 300–500 prompts（默认 500），两名人工独立标注并由盲化 LLM coder
做审计；Cohen κ 只对人工-人工计算且至少 0.70，两个主类各至少 20%，
`GLOBAL_REVISION` 覆盖至少半数 examples，`MIXED` 不超过 30%。负 sign 与
`INSERTABLE/GLOBAL_REVISION/MIXED` 满足方式标签分离；`LEX/PLAN_CORRECTNESS/`
`PLAN_COMPLETENESS/CONTEXT` 仅作附录。MIXED 超标时只增加自有 judge/rewrite
附录，不改官方 rubric/score；contradiction 未冻结前为探索性。

附件把全部负分条目并入 `GLOBAL_REVISION`；本协议把计分正负与满足方式正交，
这是有意的 class-composition 修订。主模型控制 rubric sign，并另报 negative
violation。内容 taxonomy 仍逐 item 独立标注和审定，但只进入附录。W0 的
reference/Base resource pilot 与预先标注的 rubric manifest 分开；confirmatory
interaction 只使用设置冻结后的 Full-pass completions、排除 `MIXED`。repair 的
GLOBAL_REVISION “不抬”用 1 pp 等价界的 90% TOST，不用不显著代替。

E1/E2 主表都使用同一组 11 个命名行（Base、Teacher、SFT、Vanilla OPD、
FastOPD fixed-prefix、SKD、TA-OPD、TIP-select、Relay、TRD canonical、RvI）。
repair-only、intervene-only、detached、A2 等放 supplementary/mechanism，
每一行有独立的实现合同；没有可复现上游代码时标 clean-room，不写 official
reproduction。

## 4. 编号、里程碑与主张边界

消融严格采用 A1–A8；旧 A9/A10 不再是最终编号，teacher-cost sweep 归入
A8 Pareto 附录。附件的 W0–W4 是基准顺序；`healthbench-first` 分支由上述
H0–H4→GO/STOP 顺序覆盖。预算耗尽时标记 `NOT_RUN_BUDGET_CAP`，不可按观察
结果挑替代 run；科学门失败使用不同的 `NOT_RUN_HEALTHBENCH_GATE`。

本文动机只引用 TRD 的 `g_frag/g_ideal` context 论证，不声称 support-coverage
极限或 off-support 必然不可学；不声称 HealthBench “没有任何竞品”。D3 只有
normal>detached 才支持 context 因果表述，A2 不显著则撤销状态相关性，E2 W0
失败则保留官方 Full/Hard 分而将 rubric interaction 降为探索性。
