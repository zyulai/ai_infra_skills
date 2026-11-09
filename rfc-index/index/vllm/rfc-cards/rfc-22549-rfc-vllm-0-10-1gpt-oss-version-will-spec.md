# RFC #22549: [RFC]: vllm 0.10.1gpt-oss version, will speculative sampling be used by default?

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22549](https://github.com/vllm-project/vllm/issues/22549) |
| 作者 | 123wujiao |
| 创建日期 | 2025-08-08 |
| 状态 | ⚪ closed(关闭于 2025-08-08) |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
这是一则咨询性提问(非设计提案):询问 vllm 0.10.1gpt-oss 版本是否默认启用 speculative sampling,并附 profile 推理次数异常的现象描述。

## 提案动机
作者观察到异常现象:用 VLLM 运行单请求、设定生成 100 tokens 时,profile 工具只捕获到 58 次 model inference,与「应有 100 次模型推理」的预期不符;作者怀疑与 speculative sampling 是否默认启用有关,故提问求证。正文较简短,卡片仅覆盖其字面内容。

## 提出的方案
正文无任何设计方案;「Proposed Change」一节即重复问题本身:「vllm 0.10.1gpt-oss version, will speculative sampling be used by default?」。Feedback Period、CC List、Any Other Things 三节均为「_No response_」;pack 中未包含任何答复或结论。

## 优化点与收益
正文未给出量化数字(58 次 vs 预期 100 次为现象描述,非性能收益数据)。

## 现状与落地
已关闭(2025-08-08,与创建同日),0 条评论;原帖在 pack 快照内没有讨论记录,问题是否得到解答需查站内其他渠道或关联讨论。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
