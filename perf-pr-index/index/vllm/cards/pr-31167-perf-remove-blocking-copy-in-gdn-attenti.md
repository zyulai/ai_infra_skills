# PR #31167: [Perf] Remove blocking copy in GDN Attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-12-22 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 仅改动 `vllm/v1/attention/backends/gdn_attn.py` 一行 |
| 链接 | https://github.com/vllm-project/vllm/pull/31167 |

## 一句话结论

去掉 GDN Attention metadata 构建中一处阻塞式 H2D 拷贝（改 `non_blocking=True`），解除 Qwen3-Next 上 async scheduling 无法正常工作的阻塞点，BS1/4xB200 约 +1%。

## 动机

Qwen3-Next 使用的 GDN（Gated DeltaNet）attention 后端在 `build` metadata 时，把 CPU 上的 `num_computed_tokens_cpu`（context lens）拷贝到 GPU 用的是同步 `.to(device)`——这类阻塞性主机-设备拷贝会让 async scheduling 在该后端上失效，是让 Qwen3-Next 的 async scheduling + MTP 完全非阻塞化的障碍之一。作者明确说明本 PR 的主要目的是为后续 follow-up 铺路，而非性能大改。

## 优化点

单行改动：`GDNAttentionBackend.build` 中 `context_lens_tensor = context_lens.to(query_start_loc.device, non_blocking=True)`，把同步拷贝替换为非阻塞拷贝，使该 H2D 传输不再阻塞主机线程，从而不再破坏 async scheduling 的流水。除这一处外没有任何其他逻辑改动；`nums_dict`、`batch_ptr`、`token_chunk_offset_ptr` 等其余 metadata 构建路径不变。

## 关键改动

```text
vllm/v1/attention/backends/gdn_attn.py: context_lens 的 H2D 拷贝加 non_blocking=True
```

```python
-        context_lens_tensor = context_lens.to(query_start_loc.device)
+        context_lens_tensor = context_lens.to(query_start_loc.device, non_blocking=True)
```

## 性能声明

4xB200、BS1 下 +1% speedup（PR 正文原文："Nets +1% speedup at BS1 on 4xB200"）。未给出其他吞吐/延迟数字。精度：GSM8K（5-shot）flexible-extract 0.8469±0.0099、strict-match 0.7961±0.0111（PR 未附改动前对照值）。

## 验证与回归风险

测试：GSM8K 精度跑通（数值如上）；diff 中未包含新增单测。风险面：影响所有走 GDN attention 后端的模型（Qwen3-Next 等 hybrid 线性注意力模型）的 metadata 构建路径；非阻塞拷贝的正确性依赖拷贝源的生命周期管理，PR 正文与 diff 未讨论前置条件（如源张量是否 pinned）；作者声明后续还有使 async scheduling + MTP 完全非阻塞的 follow-up，本 PR 只是解除其中一处阻塞。

## 关联

同区域（Qwen3-Next / GDN attention）：索引内另有 #24526（Qwen3-Next 模型支持）、#43534（GDN fused kernels）、#59735（GDN decode 路径）、#65874（fused GDN MTP）。PR 正文未列关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
