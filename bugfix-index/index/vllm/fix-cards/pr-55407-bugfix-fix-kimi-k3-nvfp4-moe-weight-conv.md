# PR #55407: [Bugfix] Fix Kimi K3 NVFP4 MoE weight conversion OOM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55407](https://github.com/vllm-project/vllm/pull/55407) |
| 作者 | wzhao18 |
| 合并日期 | 2026-09-07 |
| labels | bug, k3, kimi, nvidia, quantization |
| 热度 | 👍 0 · 💬 9 |
| 分类 | MoE |
| bug 类型 | resource |

## 一句话结论
Kimi K3 NVFP4 量化后处理碎片化内存下逐专家分配+stack 导致 OOM;改为预分配最终张量逐专家填充,消除峰值。

## 问题现象
运行 Kimi K3 NVFP4 时,量化后处理(quantization post processing)因内存碎片化抛出约 15000 行如下警告,部分情况下直接致命 OOM:
```
[rank6]:[W904 12:06:06.083917542 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 6 while trying to allocate 20971520 bytes (free: 16252928, total: 287428640768).
```
影响面:NVFP4 量化 MoE 权重做 FlashInfer TRT-LLM shuffle 离线转换的加载路径,权重越大、专家越多越容易触发。

## 根因
正文披露:转换过程为每个专家分配结果张量再 `torch.stack`,在 `_scoped_allocator_max_split(max_split_size_mb=20)` 下,大内存块无法被切分,逐专家结果分配失败,产生碎片化内存下的 OOM 警告乃至致命失败。即:全部专家的临时结果与最终 stack 结果同时驻留,峰值内存远超必要值,在碎片化分配器约束下无法获得连续大块。

## 修复方案
改 `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` 的 `prepare_static_weights_for_trtllm_fp4_moe`:先用 `torch.empty_like` 预分配 4 个最终 FlashInfer TRT-LLM shuffled 张量(gemm1/gemm2 的权重与 scale)及 2 个单专家 scale scratch;permute 索引只按专家 0 计算一次;循环内逐专家用 `torch.index_select(..., out=最终张量[expert_id])` 原地填充,scale 经 scratch 做 `nvfp4_block_scale_interleave` 后 `copy_` 写入;删除 list append 与 `torch.stack`。另补全函数类型标注。

```python
+    gemm1_weights_fp4_shuffled = torch.empty_like(gemm1_weights_fp4, dtype=torch.uint8)
+    gemm1_scales_fp4_shuffled = torch.empty_like(gemm1_scales_linear_fp4)
+    gemm2_weights_fp4_shuffled = torch.empty_like(gemm2_weights_fp4, dtype=torch.uint8)
+    gemm2_scales_fp4_shuffled = torch.empty_like(gemm2_scales_linear_fp4)
...
+    for expert_id in range(num_experts):
+        torch.index_select(
+            gemm1_weights_fp4[expert_id].view(torch.uint8),
+            0,
+            gemm1_weight_indices,
+            out=gemm1_weights_fp4_shuffled[expert_id],
+        )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `flashinfer_fp4_moe.py` 的 NVFP4 MoE 权重 shuffle 转换函数;输出张量形状/布局不变(同 `empty_like` 形状),数值上逐专家 `index_select` 等价于原逐专家索引重排;索引按专家 0 缓存一次与原来逐专家重算一致(各专家形状相同);TRT-LLM kernel 侧无改动。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
