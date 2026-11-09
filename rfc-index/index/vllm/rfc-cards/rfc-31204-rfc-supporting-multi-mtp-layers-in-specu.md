# RFC #31204: [RFC]: Supporting Multi MTP layers in Speculative Decoding (EagleProposer)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31204](https://github.com/vllm-project/vllm/issues/31204) |
| 作者 | DingYiBin |
| 创建日期 | 2025-12-23 |
| 状态 | ⚪ closed(关闭于 2026-07-28) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Speculative Decoding |

## 一句话结论
提议扩展 EagleProposer 支持多 MTP 层投机解码(当前仅用第一层),以覆盖拥有 3 个 MTP 层的 MiMo-V2-Flash 等模型。

## 提案动机
EagleProposer 目前只能利用模型的第一个 MTP 层,而 XiaomiMiMo/MiMo-V2-Flash 有 3 个 MTP 层,hugo-wind-ding/qwq-32b-mtp 为 QwQ-32B 提供了 7 个 MTP 层。作者询问是否有计划或正在进行的 PR 来支持投机解码中的多 MTP 层。

## 提出的方案
- EagleProposer 需新增一个成员函数,在 num_nextn_predict_layers > 1 且 num_speculative_tokens > 1 时向 MTP 模型传入 spec_step_idx
- 注:pack 正文仅包含上述动机与一句提案,无更多设计细节

## 优化点与收益
正文未给出量化数字,也未给出多 MTP 层的预期收益数据。

## 现状与落地
已关闭(2026-07-28),labels 含 stale;正文未说明是否实现,落地情况需查关联 PR。

## 关联
- XiaomiMiMo/MiMo-V2-Flash 与 hugo-wind-ding/qwq-32b-mtp(HuggingFace 模型页);正文未列出其他 RFC/PR

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
