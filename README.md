# RvI-OPD

Repair-vs-Intervene On-Policy Distillation 的可审计实验仓库。

本仓库把“应修当前位置，还是应改变后续状态”写成一组可证伪的实验合同。它包含完整预注册配置、纯 CPU 信号/路由/预算/统计核心、确定性 smoke test 和 GPU 训练接入规范；**目前没有真实训练结果，synthetic smoke 只证明软件合同成立，不能作为论文证据。**

## 已实现

- TA-OPD 口径的 top-K 并集与 `KL(T||S)`；batch q05/q95 分解保留为诊断，生产路由使用 D1 冻结的 global raw-`D/C` anchors。
- tokenizer-specific reflection-token mass 接口与 Relay-style handoff trigger。
- `repair / intervene / discard` 路由，以及默认关闭的 D4 repetition bypass、D5 `p̂=0` rescue。
- front-loading、有效 trigger cooldown、`M` 上限和实际 bridge token 配额。
- 验收门的 D1 frozen joint max-statistic artifact/evaluator、paired probe、失败回滚及“失败成本不退款”账本。
- problem-cluster paired bootstrap、difference-in-differences、Holm 校正。
- D0–D5、E1、E2 和 A1–A8 的机器可读实验配置。
- 无第三方运行时依赖的 CI/smoke 路径。

## 快速开始

```bash
git clone https://github.com/fanghouzheng/rvi-opd.git
cd rvi-opd
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-bootstrap.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
make env-check
make validate
make test
make smoke
```

smoke 产物位于 `runs/smoke/manifest.json` 和 `runs/smoke/report.json`，`runs/` 默认不提交。

`make validate` 检查当前预注册矩阵；真正启动 GPU/评测前还必须运行
`rvi-opd validate-config --config-dir configs --run-ready`。后者会拒绝尚未冻结的
template、rubric manifest 等 SHA256 占位符；因此仓库刚 clone 下来时预期不会通过，
需先在 C0/W0 生成并回填对应工件，而不能用任意字符串跳过。

## 科学执行顺序

```text
数据/许可证/泄漏审计
        ↓
 D1 信号定标并冻结阈值
        ↓
 D2 paired continuation（W1）
        ↓ 仅当 W1 通过
 D0 2×2（s2 高/低预设子群）+ D3 detached + A2 shuffled
        ↓
 D4/D5 边界实验
        ↓
 E1 数学主表 → E2 医疗主表
```

最关键的四个 confirmatory hypothesis 是：

1. D0 的 forced-action ITT 在 `D^L`/`D^I` 两个预设 signal strata 间存在正的效应异质性（`Delta2`）；signal strata 与 s2 高/低都不是随机化因子。
2. D2 中 repair 能改善被修位置的 KL，但 downstream s2 在预注册等价界内不变；bridge 同时改善 s2 与 verifier。
3. D3 中 normal bridge 优于使用相同 teacher leg、但不把它放进后续 KV/context 的 detached bridge。
4. RvI 优于保持动作数、位置、bridge 长度和成本不变的 A2 action-shuffled 对照。

任何一个机制门失败，都按 [实验计划](docs/EXPERIMENT_PLAN.zh-CN.md) 中的降级规则报告，不用主表掩盖。

## 两个必须区分的“预算”

机制表的配平声明仅限 **target/query-position count matched**：用 `teacher_scored_tokens` 对齐被查询或监督的位置数；这不等同于 compute matched。`teacher_inserted_tokens` 只存在于 intervene 臂，是动作本身的定义，不能伪装成 repair 也可匹配的成本。

`teacher_forward_calls`、prefill、decode/generation、gate 和 GPU-seconds 均只逐臂报告，所有 rejected bridge 的成本也不得退款或隐藏。详见 [方法规格](docs/METHOD_SPEC.md)。

## 训练层边界

当前提交是研究协议与无 GPU 核心，不复制 TA-OPD/TIP/PACED 等上游代码。大模型训练应在锁定的 `verl`/Relay/TRD revision 上做 clean-room adapter，并先通过同一套 audit contracts。接入点和需要保存的 artifacts 见 [复现规范](docs/REPRODUCIBILITY.md)。

环境不能混装：CPU 合同层使用 `requirements.txt`（运行时零第三方依赖）与
`requirements-dev.txt`；确认性 GPU 实验只使用 `requirements-gpu-cu130.txt`
指向的 Relay 锁定栈。独立 verl/TRD checkout 不得覆盖该环境中的 Relay fork。
系统版本、安装顺序、校验命令和常见错误见 [环境与依赖](docs/ENVIRONMENT.zh-CN.md)。

## 目录

```text
configs/                 D0–D5、E1/E2、Ablations 的冻结配置
docs/                    方法、完整实验、医疗标注、复现与决策记录
src/rvi_opd/             dependency-free 信号、路由、预算、gate、统计核心
tests/                   单元与 deterministic integration tests
examples/                不含真实 benchmark 内容的 synthetic schema 示例
.github/workflows/       CPU CI
requirements*.txt        CPU/dev 与确认性 GPU 依赖入口
environment*.yml         Conda/Mamba 基础环境
environment-lock.json    两类环境的机器可读版本合同
```

## 重要边界

- `Qwen3-1.7B-Non-Thinking` 不是单独 checkpoint；E1 使用 `Qwen/Qwen3-1.7B`，并通过项目自有 `rvi_opd_non_thinking_v1` serializer 固定非思考格式（不会把原生 `enable_thinking=false` 参数当作跨模型模板）。
- E1 主表固定七个基准（AIME24/25/26、AMC23、HMMT-Feb26、MATH500、OlympiadBench）；E2 使用 `Qwen/Qwen3-4B-Instruct-2507 → Qwen/Qwen3-0.6B`，HealthBench Full/Hard 只作评测。
- 主路由的 `s1/s2` 阈值固定为域内 D1 的 global q80；q25/q75 只定义 D0 signal cells 与 low/high 分析子群。
- TRD 报告的 epistemic mass 数值来自其特定 OPSD 诊断，不能当作本项目跨模型固定阈值；本项目只在 D1 calibration split 上定标。
- HealthBench Full 和 Hard 重叠，绝不按 6,000 个独立样本合并；仓库也不提交或打印任何 HealthBench 示例。
- DAPO-Math-17K 必须按规范化题目去重并冻结 manifest，不能信任下载后的物理行数。

## 文档入口

- [完整实验计划](docs/EXPERIMENT_PLAN.zh-CN.md)
- [环境与依赖](docs/ENVIRONMENT.zh-CN.md)
- [方法与实现规格](docs/METHOD_SPEC.md)
- [HealthBench 标注协议](docs/HEALTHBENCH_PROTOCOL.zh-CN.md)
- [复现与泄漏防护](docs/REPRODUCIBILITY.md)
- [算力与分阶段执行](docs/COMPUTE_PLAN.md)
- [预注册表格空壳](docs/TABLE_SHELLS.md)
- [关键设计决策](docs/DECISIONS.md)
- [一手资料与上游仓库](docs/REFERENCES.md)
