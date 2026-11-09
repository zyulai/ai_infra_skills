# RFC #22547: [RFC]: vllm 0.10.1gpt-oss version, will speculative sampling be used by default?

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22547](https://github.com/vllm-project/vllm/issues/22547) |
| 作者 | 123wujiao |
| 创建日期 | 2025-08-08 |
| 状态 | ⚪ closed(关闭于 2025-08-08) |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
用户提问帖:询问 vllm 0.10.1gpt-oss 版本是否默认启用推测采样,并附一个 profile 计数异常观察。

## 提案动机
作者在运行 pre-release 的 vllm 0.10.1gpt-oss 版本时观察到异常现象:单请求设置生成 100 tokens,profile 工具只捕获到 58 次模型推理,与"应有 100 次模型推理"的预期不符,由此推测该版本可能默认启用了 speculative sampling,故发帖求证。

## 提出的方案
正文未提出任何设计、接口或改动;本质是使用行为咨询:
- 核心问题:vllm 0.10.1gpt-oss 版本是否默认使用 speculative sampling?
- 支撑观察:生成 100 tokens 时 profile 仅见 58 次模型推理过程。
Feedback Period、CC List、Any Other Things 均为"No response"。pack 收录正文较短,且无 maintainer 回复内容(0 条评论),该行为问题在此 pack 内未获得解答。

## 优化点与收益
正文未给出量化数字(100 tokens 对 58 次推理是行为观察,不是性能收益主张)。

## 现状与落地
已关闭 —— 创建当日(2025-08-08)即关闭,0 条评论;正文与评论均看不出问题是否得到解答,结论需查原帖或关联讨论。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
