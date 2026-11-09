# PR #9477: Optimize moe_sum_reduce_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-09-07 |
| 分类 | Attention / Kernel (tier: major) |
| Label | (无) |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 benchmark 脚本与 fused_moe_triton_kernels.py 中同 kernel 的两份实现 |
| 链接 | https://github.com/sgl-project/sglang/pull/9477 |

## 一句话结论

把 TP MoE prefill 阶段占比很高的 `moe_sum_reduce_kernel` 从逐 token 循环改为 2D tile 化累加并加大 num_warps,kernel 提速最高 17.6%,DeepSeek-V3 E2E 吞吐提升 2–5%。

## 动机

profiling 显示 TP MoE 的 prefill 阶段中 `moe_sum_reduce_kernel`(把 topk 个专家输出按 routed_scaling_factor 归约回单份 hidden state)占比较高,值得专项优化。原 kernel 按 token 一维循环、每个 token 内对 topk 维循环累加,一次只处理 BLOCK_DIM=2048 一维切片,并行度和访存效率都不足。

## 优化点

kernel 重写为 2D tile 化:一次程序处理 `[BLOCK_M, BLOCK_DIM]` 的 token x 维度 tile,用 `offs_token`/`offs_dim` + 掩码取代原先的 `token_start/token_end` 与 `dim_start/dim_end` 循环边界计算;对 topk 维的循环保持 `tl.range(..., num_stages=NUM_STAGE)` 流水线,但累加器升为二维 `tl.zeros((BLOCK_M, BLOCK_DIM))`,load/store 均按 tile 指针一次性完成,减少了逐 token 的地址计算与循环开销。同时 `num_warps` 从 8 提高到 16。BLOCK_M=1、BLOCK_DIM=2048 的启动参数保持不变(实际 BLOCK_M=1 时收益主要来自消除外层 Python 端 token 循环与写回合并)。同一 kernel 在 `benchmark_sum_scale.py` 与 `fused_moe_triton_kernels.py` 两处同步修改。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py: _moe_sum_reduce_kernel 改为 [BLOCK_M, BLOCK_DIM] 2D tile 累加,num_warps 8→16
benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py: 同步修改基准脚本中的 kernel 副本
```

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
    tile = tl.load(base_ptrs + i * input_stride_1,
                   mask=mask_token[:, None] & mask_dim[None, :], other=0.0)
    accumulator += tile.to(tl.float32)
accumulator *= routed_scaling_factor
```

## 性能声明

kernel 级 benchmark(`sum_scaled_performance`,正/反向数值为 do_bench 单位,正文未注明硬件):num_tokens 4096 时 TritonKernel 从 104.032 提升到 109.664、小 token 段(1–128)从 7.84–10.688 提升到 6.688–10.016;作者总结 kernel speedup **最高 17.6%**,两种实现均通过 "All implementations match" 正确性校验。

端到端(DeepSeek-V3,TP8,`--mem-fraction-static 0.9`,gsm8k 200 题 parallel 128 num-shots 8):第二轮起吞吐提升 **10%**——本 PR Output throughput **1166.097 token/s** / Latency 17.525 s,Main 为 **1033.465 token/s** / 19.708 s,Accuracy 0.970 vs 0.965(第一轮 warmup 两边都不高,300.540 token/s)。作者总结 **DeepSeek-V3 E2E 吞吐 speedup 2–5%**。

## 验证与回归风险

验证依赖 benchmark 脚本内置的正确性比对(两种实现输出一致)。回归风险:`num_warps=16` 与固定 BLOCK_M=1/BLOCK_DIM=2048 未按硬件/shape 自适应,极小 token 数下 kernel 数值上仍略优于原版但 TorchCompile 在 32 token 以上反超;精度差异(gsm8k 0.970 vs 0.965)作者未解释是否波动;影响所有走 fused MoE Triton 路径的模型(TP MoE prefill/decode)。

## 关联

同区域:fused MoE Triton kernel 系列(如 moe_align、topk 等);pack 未列出具体关联 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
