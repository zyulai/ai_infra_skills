# RFC #17140: [RFC]: Native support for Mamba, SSM, and hybrid transformer models in vLLM V1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17140](https://github.com/vllm-project/vllm/issues/17140) |
| 作者 | tlrmchlsmth |
| 创建日期 | 2025-04-24 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 11 · 💬 7 |
| 分类 | Architecture / Engine |

## 一句话结论
提议让 Mamba/SSM/混合模型原生支持 vLLM V1:依托 Hybrid Allocator 与 torch.compile 补齐,兼作长期跟踪清单。

## 提案动机
Mamba、SSM 与混合 transformer 模型是走向「随序列长度线性扩展」的重要路径;vLLM 已支持该类多个模型(Jamba、Mamba、Codestral Mamba、Falcon Mamba、Bamba、Zamba2、Plamo2,各支持 PR 见正文链接),应继续保持优秀支持。问题:SSM 模型在 vLLM 中的支持普遍弱于 transformer,存在多项缺陷——不被 vLLM V1 支持,且应在 V0 弃用之前支持;SSM 状态管理较 hacky、由模型定义自行管理;SSM 状态不经 block manager 管理,导致与 prefix caching、KV cache offloading、prefill-decode 分离不兼容;chunked prefill 存在严重性能问题。本 RFC 提出若干改进(部分已在进行中)并兼作 issue 跟踪器。

## 提出的方案
V1 支持的 blocker 清单:
- Hybrid Allocator(#11382):已完成初始工作(初始目标面向 sliding-window attention)
- Hybrid Allocator 落地后,扩展它以支持 SSM 与混合模型(待办)
- torch.compile 支持(piecewise CUDA graphs 需要)(待办)

其他改进:
- 将 Mamba 支持扩展到 CUDA GPU 之外(待办)
- 改进 chunked prefill 性能:已完成,经 PR #16942、PR #17146
- Mamba2 的量化 + 张量并行支持(#14618)(待办)

## 优化点与收益
正文未给出量化数字(定性:解决 chunked prefill 性能问题、补齐 V1 支持,打通 prefix caching/KV offloading/PD 分离等依赖 block manager 的能力)。

## 现状与落地
open(labels 含 keep-open),作为长期跟踪 issue 保留;blocker 中 Hybrid Allocator 已完成,SSM/混合模型扩展与 torch.compile 支持仍待办。

## 关联
- Hybrid Allocator:https://github.com/vllm-project/vllm/issues/11382
- Mamba2 量化+TP:https://github.com/vllm-project/vllm/issues/14618
- PR #16942:https://github.com/vllm-project/vllm/pull/16942
- PR #17146:https://github.com/vllm-project/vllm/pull/17146
- 另有 7 个模型支持 PR 链接(Jamba/Mamba/Codestral Mamba/Falcon Mamba/Bamba/Zamba2/Plamo2)见正文

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
