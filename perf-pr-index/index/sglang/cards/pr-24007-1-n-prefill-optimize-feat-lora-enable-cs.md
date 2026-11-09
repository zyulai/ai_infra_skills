# PR #24007: (1/n - prefill optimize)feat(lora): enable csgmv backend with virtual experts for MoE LoRA

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yushengsu-thu |
| 合并 | 2026-05-04 |
| 分类 | MoE (tier: major) |
| Label | high priority, lora, run-ci |
| Diff | pack 未提供 diffstat（diff 内容可用：4 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/24007 |

## 一句话结论

解除 MoE LoRA virtual experts 只准配 triton backend 的限制并修复 CUDA graph 崩溃，csgmv 后端在 MoE LoRA 下吞吐反超 triton 约 1.7-2.2%。

## 动机

MoE LoRA 的 virtual experts 路径用自有的 triton kernel `merged_experts_fused_moe_lora_add`，与 dense-LoRA 后端无关，但 `server_args.py` 里有一条 assert 把 `--lora-use-virtual-experts` 硬性限制在 `--lora-backend triton`，挡住了 csgmv 后端组合。此外直接组合会因 csgmv 的 chunked-segment 语义与 MoE virtual experts 期望的 per-request 语义不一致，产生错误的 token-to-adapter 映射并触发 device-side assert，CUDA graph capture 即崩溃。

## 优化点

机制上是"兼容性打通 + 语义对齐"：1) 删除 triton-only 断言；2) `LoRABatchInfo` 新增 per-request 的 `req_seg_indptr`（每请求累计 token 边界）与 `req_weight_indices`（每请求 adapter 索引）两个字段，csgmv 后端在 `prepare_lora_batch` 里单独计算并预分配（CUDA graph 场景预置 buffer），decode 用 `arange`、非 decode 用 CPU 端 `cumsum` 构造；3) MoE LoRA 的 `_get_lora_info` 优先取 per-request 字段，fallback 到原 `seg_indptr/weight_indices`。triton 后端两组字段本就相同，因此 triton 路径行为不变；csgmv 的内部 chunk 分段不再泄漏给 MoE virtual experts。

## 关键改动

```text
python/sglang/srt/server_args.py: 删除 --lora-use-virtual-experts 的 triton-only assert
python/sglang/srt/lora/utils.py: LoRABatchInfo 增加 req_seg_indptr / req_weight_indices 字段
python/sglang/srt/lora/backend/chunked_backend.py: csgmv 后端计算并存储 per-request segment 信息（含 CUDA graph 预分配）
python/sglang/srt/lora/layers.py: MoE LoRA _get_lora_info 优先使用 per-request 字段
```

## 性能声明

Qwen3-30B-A3B，TP=4，GB300，in=1024 / out=2048，e2e tps（triton vs csgmv）：BS=1 201.0/s vs 205.5/s（102.2%）；BS=128 2753.3/s vs 2800.3/s（101.7%）；BS=512 2758.7/s vs 2807.6/s（101.8%）。即 csgmv 快约 1.7-2.2%——PR 解释为 MoE LoRA 无论 dense-LoRA 后端选谁都走同一个 virtual experts triton kernel，差距来自后端侧开销。准确度：KL(sglang, trainer)=0.00392，低于 KL(orig_sampler, trainer)=0.00424 与 KL(sglang, orig_sampler)=0.00497。

## 验证与回归风险

测试计划覆盖：triton vs csgmv 性能对比（BS=1/128/512）、对训练 logprobs 的 KL 散度验证、csgmv + virtual experts 的 CUDA graph capture 成功。风险：新增 per-request 字段只在 csgmv+virtual-experts 组合生效，但 `_get_lora_info` 改为条件选择字段，若某些路径漏填新字段会走 fallback（行为不变）；CUDA graph 预分配 buffer 尺寸绑定 `max_bs_in_cuda_graph`，超大 batch 需确认截断逻辑（PR 未展开）。

## 关联

标题标注 "1/n - prefill optimize"，为 prefill 优化系列第 1 篇（后续 PR 未在正文列出）；同分类 MoE 其他相关 PR 本卡未及，参见索引同区条目。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
