# HealthBench 评测与 rubric 满足方式标注协议

## 1. 隔离原则

HealthBench 仅用于最终评测。prompt、rubric、example、reference completion、grader rationale 不能进入训练、D1、gate、prompt 开发或人工示例文档。仓库不提交或打印任何真实条目。评测数据锁定为 `openai/healthbench@40ee1968852fc57f625934251ac22be47077a8fb`（或该快照中与 simple-evals 对应的 JSONL 文件），运行前记录文件 SHA256；不得以 `main`、下载当天的 URL 内容或未记录的镜像替代。

Full 有 5,000 prompts，Hard 是其中 1,000 个的子集。当前对应 JSONL 文件的 SHA256 为 Full `e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f`、Hard `b0320430e5cd974e746585594c1dd10b5a3fc2aff9c72b26106c2c4a069d74e9`；若上游快照变更必须重新核验并更新 lock。Hard→Full 映射的冻结序列化为“Hard 文件顺序、每行 UTF-8 紧凑 JSON `{prompt_id,full_index}`、LF 换行”，当前 SHA256 为 `64852846390fa7b3f65e1f0ae93d0160318188b6264023673184eef2dcf7bca7`。运行前只比较并保存 prompt-ID 集合及其索引映射的 hash（不保存题目文本）；必须验证 `Hard ⊂ Full` 且 ID 唯一。对 Full 生成一次 completion 后按 IDs 索引 Hard；不为 Hard 独立重采样，也不能将两者相加后按 6,000 个独立 prompt 做统计。

## 2. 官方分数

使用 `openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47` 的原始 physician grader template 和 aggregation：grader 固定 `gpt-4.1-2025-04-14`、`max_tokens=2048`、主分析 `n_repeats=1`，不做本地 sampler/template override。simple-evals 的 CLI 默认面向其内置 API 模型；本地 Qwen checkpoint 必须先由项目 adapter 生成 completion manifest，再以同一 prompt/completion 记录调用官方 `HealthBenchEval` grader，不能把 `--model=gpt-4.1` 的 grader 结果误当作 Qwen 的回答。每个 example 的 raw score 为“所有 satisfied items 的 weights（包括负权重）之和 / 正权重之和”；**不逐 example clip**，而是先对 prompt raw scores 取均值，再把该 aggregate clip 到 `[0,1]`。不重写 rubric item、不修改权重、不把辅助分类放回官方分数。

负分条目单独报告 `sum(abs(weight)×violation_indicator)/sum(abs(weight))` 的 weighted violation rate。若只保留负分条目导致官方正分分母为 0，不计算伪“官方 normalized score”。

在输出前预注册 10% prompt sample，对其额外做 3 次独立 regrade 并报告一致性/方差；该 audit 不替换 `n_repeats=1` 的官方主分析。每次 grader call 都计入预算，cache key 至少含 `(prompt_id, completion_hash, rubric_id, grader_model, grader_revision, template_hash, repeat_index)`；重试不得覆盖原始响应或改变 repeat index。

### 模型、serializer 与生成合同

E2 teacher 固定为 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，student 固定为 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。student 是由 `Qwen3-0.6B-Base` 衍生的 post-trained hybrid checkpoint；`base_model` lineage 不等于纯 Base，也不能改称 `Qwen3-0.6B-Instruct`。两者均为 `Qwen3ForCausalLM`，但官方 tokenizer 配置和 chat template 不同；lock 必须保存各文件 hash、特殊 token 映射、`config.vocab_size` 与 tokenizer length。

所有 arm 使用项目自有的 `rvi_opd_non_thinking_v1` serializer；E2 system message 固定为 `You are a helpful medical assistant. Provide a clear, complete, and safe answer while stating uncertainty and escalation conditions when relevant.`，该字符串必须纳入 serializer/rendered-prompt hash。并在 C0 冻结 teacher/student 的 rendered token IDs、prompt hash、`Complex_CoT`/`Response` target 映射和 loss mask。不能直接把 student 官方 `enable_thinking=false` 当作共同模板：它会插入空的 `<think>...</think>` 块，而 teacher 模板不会。若输出含 `<think>`，按冻结的 parser/违规策略处理，禁止在看到结果后静默剥离。

回答生成不继承任一模型的 `generation_config.json`，而是显式传 `do_sample=true`、`temperature=1.0`、`top_p=1.0`、禁用 `top_k`、stop IDs `[151645, 151643]`、`pad_token_id=151643`，每个 checkpoint/prompt 只生成一条 completion。项目配置以 `top_k=0` 表示禁用；Relay/verl adapter 在 dispatch 前映射为其后端值 `-1`，并同时记录项目值和实际请求值。每条请求先在渲染后的 token IDs 上检查：

```text
rendered_input_tokens + requested_new_tokens + eos_reserve <= runtime_context_limit
```

本协议把 stop token 计入后端的 `max_new_tokens`，因此默认 `eos_reserve=0`；若后端另需预留 EOS，必须在调用前记录正值并相应降低回答 cap。

协议中的运行时上限是 `max_model_length=40,960`、输入 cap `36,864`、回答 cap `4,096`；若任一上限不满足，必须在调用前失败并记账，不得由 vLLM/verl 静默截断或 clamp。Qwen 文档将 32,768 视为预训练上下文、40,960 视为典型运行时分配；使用超过 32,768 的输入要单独标注为长上下文扩展并通过 C0 长度分层检查。teacher/student/arm 共用同一 prompt 与 sampling manifest。

## 3. 辅助分类

在 Full 中按 Hard membership 与冻结的官方 theme 分层抽取 **500 prompts**（seed `20260828`；允许在任何模型输出可见前预注册地降至 300），对入样 prompt 的全部 rubric items 标注；这是模型输出前冻结的 annotation manifest，不是 resource/grader pilot 的 arm outcome。sample manifest 在任何模型输出可见前冻结。标注者只看 conversation 与 rubric，不看任何模型输出、方法名或分数。

- `INSERTABLE`：一个局部、可独立插入的片段足以满足，且不要求改写其他部分。
- `GLOBAL_REVISION`：是否满足依赖整体方案、跨段一致性、全文无冲突、行动优先级或全局安全性。
- `MIXED`：局部片段可能有帮助，但无法在不看全文结构的情况下稳定判断。

负分条目不自动归入 `GLOBAL_REVISION`；label 与 rubric sign 分开记录，避免混淆“满足方式”和“正负得分”。

内容类型 `LEX`、`PLAN_CORRECTNESS`、`PLAN_COMPLETENESS`、`CONTEXT` 是每个入样 item 的第二个独立标签，按同一盲化/审定/哈希流程冻结，但只作附录归因，不能替代上述主轴或进入 W0 gate。若一个条目同时包含局部和全局要求，标为 `MIXED`；只有在 W0 预注册的 MIXED 比例超过 30% 时，才启动独立的自有 judge/rewrite 附录，不回写官方 rubric 或官方分数。自相矛盾检测也属于 rubric 外的探索性指标；另描述性报告“泛化安全/升级话术存在、但缺少问题特定可执行方案完整性”的 boilerplate rate。

## 4. W0 前置门

- 两位标注者独立标注，Cohen's κ ≥ 0.70。
- 两位人工标注者独立标注，另有盲化 LLM coder 做第三份编码和 disagreement audit；W0 的 κ 门只计算人工-人工，LLM 与人工的一致性另报，不能替代人工门。
- 在冻结前确认官方 theme、rubric weight、rubric sign 及（如存在）官方 axis 字段的可用性；官方字段校验不等于自建满足方式标签。
- 在 adjudication 后的全部 rubric items 中，INSERTABLE 和 GLOBAL_REVISION 各占至少 20%，并公开 item-level 各类数量。
- 在预注册的 500-prompt 标注样本中（若事先冻结为 300，则以 300 为分母），至少 50% examples 含一个或更多 GLOBAL_REVISION rubric item，并公开 example-level coverage 分子/分母。
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

标注文件必须在模型输出文件之前完成并生成 SHA256。另为每个 completion 保存只含 metadata 的 manifest（prompt ID、模型/revision、serializer/template hash、sampling 参数、输入/输出 token 数、completion hash、状态和错误码）；Full→Hard index manifest 只保存 ID/hash。论文公开 hash、统计摘要和允许发布的派生标签；遵守 HealthBench 的反泄漏要求，不公开题目文本、rubric、completion 或 grader rationale。

## 6. 分析

- 先冻结 500-prompt rubric sample，完成盲化标注、裁定并生成 annotation hash；在此之后才允许用一个锁定的 reference/Base checkpoint 在该冻结样本上运行 resource/grader pilot（只用于长度、成本、延迟、grader 一致性和预算估计；不得比较候选 arms 或调 router/loss/gate/claim）。设置、serializer hash 和预算冻结后，再用独立 sampling manifest 对全部 5,000 Full prompts 做一次正式生成；pilot completions 不进入终表。Full official score 是 confirmatory；Hard 是从同一批 Full completions 按 ID 索引的 key secondary。训练臂先重采样 3 个 fixed training seeds，再在其内做 prompt-cluster paired bootstrap 并报 leave-one-seed-out；两套 benchmark 不视为独立样本集。
- rubric interaction：mixed model 控制 rubric weight、rubric sign，rubric item 嵌套于 response/prompt。
- rubric interaction 只使用冻结 annotation manifest 中的最终 Full-pass completions，并排除 MIXED。主 contrast 明确为 `[(RvI−repair)_GLOBAL_REVISION]−[(RvI−repair)_INSERTABLE]`，方向为 Holm-adjusted 95% CI 下界 >0；`intervene-only` 的同型 contrast 是同一 Holm family 内的次要检验。另检验 repair/intervene 相对 Base 的 INSERTABLE 正增益、intervene 的 GLOBAL_REVISION 正增益，并对 repair 的 GLOBAL_REVISION 变化做绝对 0.01 界的 90% TOST；“不显著”不是“无实质变化”。不得用 post-hoc 的 `intervene OR RvI` 选择较大者。
- 另报排除负分条目后的类型 interaction，以及所有负分条目的 weighted violation rate。
- contradiction 一致性指标在独立 NLI/LLM judge 的 immutable revision、template、阈值与盲化人工复核规则冻结前仅为探索性。
- Teacher 不得同时担任 evaluator。
- 报告实际 grader calls（每个 rubric item、主重复和 audit 重复分别计数）、失败/重试数、API 成本和缓存命中率；预算达到上限时停止新增 run 并标记 `NOT_RUN_BUDGET_CAP`。

核心预测是 repair 与 intervene 都可能提高 INSERTABLE，但 RvI 相对 repair 对 GLOBAL_REVISION 的增益更大。若该 interaction 不成立，仍可报告官方总分，但不能用医疗域支持“cosmetic correction”机制。无论结果如何，本实验只评估 HealthBench external-domain behavior，不证明临床有效性、患者安全或可部署性。
