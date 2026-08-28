# HealthBench 评测与 rubric 满足方式标注协议

## 1. 隔离原则

HealthBench 仅用于最终评测。prompt、rubric、example、reference completion、grader rationale 不能进入训练、D1、gate、prompt 开发或人工示例文档。仓库不提交或打印任何真实条目。

Full 有 5,000 prompts，Hard 是其中 1,000 个的子集。对 Full 生成一次 completion 后按 IDs 索引 Hard；不为 Hard 独立重采样，也不能将两者相加后按 6,000 个独立 prompt 做统计。

## 2. 官方分数

使用 `openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47` 的原始数据、physician grader template 和 aggregation：grader 固定 `gpt-4.1-2025-04-14`、`max_tokens=2048`、主分析 `n_repeats=1`，不做本地 sampler/template override。每个 example 的 raw score 为“所有 satisfied items 的 weights（包括负权重）之和 / 正权重之和”；**不逐 example clip**，而是先对 prompt raw scores 取均值，再把该 aggregate clip 到 `[0,1]`。不重写 rubric item、不修改权重、不把辅助分类放回官方分数。

负分条目单独报告 `sum(abs(weight)×violation_indicator)/sum(abs(weight))` 的 weighted violation rate。若只保留负分条目导致官方正分分母为 0，不计算伪“官方 normalized score”。

在输出前预注册 10% prompt sample，对其额外做 3 次独立 regrade 并报告一致性/方差；该 audit 不替换 `n_repeats=1` 的官方主分析。cache key 至少含 `(prompt_id, completion_hash, rubric_id, grader_model, template_hash, repeat_index)`。

## 3. 辅助分类

在 Full 中按 Hard membership 与冻结的官方 theme 分层抽取 1,000 prompts（seed `20260828`），对入样 prompt 的全部 rubric items 标注；sample manifest 在任何模型输出可见前冻结。标注者只看 conversation 与 rubric，不看任何模型输出、方法名或分数。

- `INSERTABLE`：一个局部、可独立插入的片段足以满足，且不要求改写其他部分。
- `GLOBAL_REVISION`：是否满足依赖整体方案、跨段一致性、全文无冲突、行动优先级或全局安全性。
- `MIXED`：局部片段可能有帮助，但无法在不看全文结构的情况下稳定判断。

负分条目不自动归入 `GLOBAL_REVISION`；label 与 rubric sign 分开记录，避免混淆“满足方式”和“正负得分”。

## 4. W0 前置门

- 两位标注者独立标注，Cohen's κ ≥ 0.70。
- 在 adjudication 后的全部 rubric items 中，INSERTABLE 和 GLOBAL_REVISION 各占至少 20%，并公开 item-level 各类数量。
- 在预注册的 1,000-prompt 标注样本中，至少 50% examples 含一个或更多 GLOBAL_REVISION rubric item，并公开 example-level coverage 分子/分母。
- MIXED ≤ 30%。
- 对负分条目抽查官方 grader 的 satisfied/violated 语义和聚合行为。

两位标注者的全部分歧由第三位 adjudicator 在模型输出可见前裁定并冻结 hash。任何一项未过：保留官方 Full/Hard 总分，分类 interaction 降为探索性，不调 label 定义追结果。

## 5. 标注产物

每项保存：

```text
example_hash, rubric_item_hash, blinded_annotator_id,
label, confidence, short_reason_code, rubric_sign, rubric_weight,
adjudication_label, adjudicator_id, annotation_timestamp
```

标注文件必须在模型输出文件之前完成并生成 SHA256。论文公开 hash、统计摘要和允许发布的派生标签；遵守 HealthBench 的反泄漏要求，不公开题目文本。

## 6. 分析

- Full official score 是 confirmatory；Hard 是重叠子集的 key secondary。训练臂先重采样 3 个 fixed training seeds，再在其内做 prompt-cluster paired bootstrap并报 leave-one-seed-out；两套 benchmark 不视为独立样本集。
- rubric interaction：mixed model 控制 rubric weight、rubric sign，rubric item 嵌套于 response/prompt。
- 主 contrast 明确为 `[(RvI−repair)_GLOBAL_REVISION]−[(RvI−repair)_INSERTABLE]`，方向为 Holm-adjusted 95% CI 下界 >0；`intervene-only` 的同型 contrast 是同一 Holm family 内的次要检验。不得用 post-hoc 的 `intervene OR RvI` 选择较大者。
- 另报排除负分条目后的类型 interaction，以及所有负分条目的 weighted violation rate。
- contradiction 一致性指标在独立 NLI/LLM judge 的 immutable revision、template、阈值与盲化人工复核规则冻结前仅为探索性。
- Teacher 不得同时担任 evaluator。

核心预测是 repair 与 intervene 都可能提高 INSERTABLE，但 RvI 相对 repair 对 GLOBAL_REVISION 的增益更大。若该 interaction 不成立，仍可报告官方总分，但不能用医疗域支持“cosmetic correction”机制。无论结果如何，本实验只评估 HealthBench external-domain behavior，不证明临床有效性、患者安全或可部署性。
