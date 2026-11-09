# RFC #22550: [RFC]: vllm 0.10.1gpt-oss，speculative sampling be use？

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22550](https://github.com/vllm-project/vllm/issues/22550) |
| 作者 | 123wujiao |
| 创建日期 | 2025-08-08 |
| 状态 | ⚪ closed (2025-08-08) |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
实为提问帖:询问 vllm 0.10.1gpt-oss 是否默认启用投机采样,并报告 profile 推理次数与预期不符的现象。

## 提案动机
作者运行单请求生成 100 tokens,profile 工具只捕获到 58 次模型推理,与「应有 100 次模型推理」的预期不符,因此怀疑 pre-release 的 vllm 0.10.1gpt-oss 版本默认使用了投机采样,发帖求证。注:pack 正文较短,实为一个使用咨询而非设计提案,无更多内容可引。

## 提出的方案
正文未提出具体方案或设计,仅在 Proposed Change 一节重复同一问题:「vllm 0.10.1gpt-oss 版本,投机采样是否默认启用?」

## 优化点与收益
正文未给出量化数字;「100 tokens 只见 58 次推理」是观察到的现象,非基准测试结果。

## 现状与落地
已关闭(2025-08-08,与创建同日),0 条评论;正文与结尾均看不出官方结论或后续处理,落地情况需查关联 PR/讨论。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
