# RFC #31567: [RFC]: Why custom_mask is not exposed on FlashInfer to get more flexible use case?

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31567](https://github.com/vllm-project/vllm/issues/31567) |
| 作者 | npuichigo |
| 创建日期 | 2025-12-31 |
| 状态 | ⚪ closed(关闭于 2026-05-01) |
| labels | RFC, stale |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议向 FlashInfer attention 后端暴露 custom_mask,以支持相对注意力偏置等更灵活的用例。

## 提案动机
作者希望像 TensorRT-LLM 的 flashinfer attention 后端那样(正文引用了其代码位置链接)在 vLLM 的 FlashInfer 路径上获得同样的能力。**正文极短**:Motivation 仅此一句类比,未展开具体受阻场景、错误现象或需求细节。

## 提出的方案
正文 Proposed Change 仅一句话:"expose the custom_weight to support use case like relative attention bias"——暴露 custom mask 以支持 relative attention bias 一类用例。注:此处原文写作 `custom_weight`,与标题中的 `custom_mask` 表述不一致,卡片按原样照抄不加解读。正文未给出接口设计、实现细节或权衡分析。

## 优化点与收益
正文未给出量化数字,也未展开收益分析;可推断的收益仅限正文一句:让 FlashInfer 后端支持如 relative attention bias 的灵活用例,对齐 TensorRT-LLM 的能力面。

## 现状与落地
已关闭(关闭于 2026-05-01,labels 含 stale)。正文的 Feedback Period、CC List、Any Other Things 均为 _No response_,无讨论结论或实现信息,落地情况需查关联 PR。

## 关联
- [NVIDIA/TensorRT-LLM flashinfer.py](https://github.com/NVIDIA/TensorRT-LLM/blob/6c1abf2d45c77d04121ebe10f6b29abf89373c60/tensorrt_llm/_torch/attention_backend/flashinfer.py#L411C17-L411C28)(正文引用的类比实现)
- 正文未列出其他 RFC/PR/文档链接

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
