# RFC #12249: [RFC]: Hidden states processor

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#12249](https://github.com/vllm-project/vllm/issues/12249) |
| 作者 | DarkLight1337 |
| 创建日期 | 2025-01-21 |
| 状态 | ⚪ closed (关闭于 2025-09-01) |
| labels | RFC |
| 热度 | 👍 16 · 💬 21 |
| 分类 | Other |

## 一句话结论
允许用户通过 SamplingParams 传入自定义 HiddenStatesProcessor,灵活后处理模型隐藏状态并随输出返回。

## 提案动机
自 #10674 起,vLLM 用 Pooler 从模型提取 hidden states 并转换为 embedding、分类概率等,但仍不够用户友好:生成式与 pooling 模型使用不同的 model runner,导致"随生成文本一并返回 hidden states"的需求(#6165、#11397、#11577、#11606、#11905)难以实现;基于下游任务设置默认 Pooler 只覆盖常见场景,常需不直观的 `--override-pooler-config`(#12085),且缺乏对 hidden states 的自定义处理(#11065、#11881、#12162)。

## 提出的方案
- 仿照 `LogitsProcessor`(#1469),允许在 `SamplingParams` 与 `PoolingParams` 中传入自定义 `HiddenStatesProcessor`,对 hidden states 做后处理并在输出中返回;每次 `llm.generate` 可用不同 processor
- 接口为 Protocol:`__call__(model, hidden_states) -> R`,与 `compute_logits`/`pooler` 类似
- 各下游任务的默认 pooler 改为内置 HiddenStatesProcessor:Identity(奖励模型)、Normalize(prompt embedding)、Softmax(分类)、Step(PRM 模型);现有 pooling API(`LLM.encode`/`embed`/`score` 等)自动更新
- 可引用模型 LM head 的 processor 从 hidden states 计算 logits;也可扩展多模态输出(如 ImageFromHiddenStatesProcessor 经 image decoder 生成 PIL 图像,作者注明多模态输出仍有讨论空间)
- 待解决问题:TP/PP 下如何正确处理;processor 启动时未知、被排除在模型 profiling 之外,可能引发 OOM
- 附带建议:常规 model runner 也能返回 hidden states,可考虑在 V1 中合并 PoolingModelRunner 与 ModelRunner(#8779)

## 优化点与收益
正文未给出量化数字。定性收益:同一模型可服务不同下游任务、生成与隐藏状态返回统一到常规 runner、减少 Pooler 配置的复杂性。

## 现状与落地
已关闭。正文为设计提案并列出待解决问题,未写明实现结论,落地情况需查关联 PR。

## 关联
- #10674
- #1469
- #8779
- #12085

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
