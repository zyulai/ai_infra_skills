# RFC #38069: [RFC]: Can we implement n-gram and suffix speculative decoding in model_runner_v2?

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38069](https://github.com/vllm-project/vllm/issues/38069) |
| 作者 | lio1226 |
| 创建日期 | 2026-03-25 |
| 状态 | ⚪ closed(关闭于 2026-07-27) |
| labels | RFC, stale |
| 热度 | 👍 3 · 💬 5 |
| 分类 | Speculative Decoding |

## 一句话结论
询问能否在 model_runner_v2 中实现 n-gram 与 suffix 投机解码这两种免模型的投机推理方法。

## 提案动机
作者注意到 Eagle 已在 model_runner_v2 中实现,遂提议能否把 n-gram 与 suffix 两种投机推理方法也引入 v2;与基于模型的方法相比,这两种方法实现起来应当更简单。注:该 RFC 正文本身非常简短,动机与方案均只有寥寥数句,未展开技术细节。

## 提出的方案
正文方案仅一句话:基于 model_runner_v2 实现 n-gram 与 suffix speculative decoding。未给出接口设计、实现步骤或代码示例;具体如何复用 v2 的投机解码基础设施从正文无法得知。

## 优化点与收益
预期收益(按正文表述):相比基于模型的方法,n-gram 与 suffix 投机解码实现更简单,可低成本为 v2 增加免模型投机解码能力。正文未给出量化数字。

## 现状与落地
已关闭(2026-07-27)。反馈期为「初始反馈 1-2 周」,CC 了 WoosukKwon、TheEpicDolphin;社区结论与是否实际实现从正文无法看出,落地情况需查关联 PR。

## 关联
正文未列出(正文仅以文字提及 Eagle 已在 model_runner_v2 实现,未附链接)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
