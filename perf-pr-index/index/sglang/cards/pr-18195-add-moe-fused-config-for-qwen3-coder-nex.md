# PR #18195: Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-02-04 |
| 分类 | MoE (tier: major) |
| Label | 无 |
| Diff | pack 未提供 diffstat（diff 内容可用：仅新增 1 个 JSON 配置文件，146 行） |
| 链接 | https://github.com/sgl-project/sglang/pull/18195 |

## 一句话结论

为 Qwen3-Coder-Next-FP8 在 H100 TP=2 下新增 tuned Triton MoE kernel 配置，serving 吞吐 +2.2%、中位 TTFT -40.8%。

## 动机

SGLang 的 fused MoE Triton kernel 按 (E, N, dtype, block_shape, device) 查找预调优配置文件；缺少对应组合时只能回退默认 tile 参数，对 Qwen3-Coder-Next-FP8（E=512, N=256, fp8_w8a8, block_shape=[128,128]）在 H100 80GB 上没有现成配置，MoE GEMM 未按最优 tile/warp/stage 执行。本 PR 补齐这份离线调优产物。

## 优化点

不改变任何 kernel 代码，纯配置新增：为 `E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json` 提供按 token 数（batch size 1 到 4096）分档的 19 组 tile 配置。小 batch（1-96）取 BLOCK_SIZE_M=16、BLOCK_SIZE_K=128、num_warps=4、num_stages 2-4；batch 64/128/256/512 在 K 维换到 256；大 batch（2048+）BLOCK_SIZE_M 提到 64、GROUP_SIZE_M 提到 16/32、num_stages=4，即随 batch 增大逐步偏向更大 M tile 与更多 L2 分组。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 19 档 (batch size -> BLOCK_M/N/K, GROUP_SIZE_M, num_warps, num_stages) 调优配置
```

## 性能声明

`bench_serving --backend sglang-oai-chat --model Qwen/Qwen3-Coder-Next-FP8 --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 8`（硬件未注明，按配置文件推断为 H100 80GB TP=2）：Output token throughput 382.19 -> 390.63 tok/s（+2.2%）；Total token throughput 750.65 -> 767.24 tok/s（+2.2%）；Peak output token throughput 909 -> 975 tok/s（+7.3%）；Median TTFT 571.22 -> 338.33 ms（-40.8%）；P99 E2E Latency 27312.44 -> 24693.58 ms（-9.6%）；Median ITL 8.99 -> 8.25 ms（-8.2%）。

## 验证与回归风险

PR 未提供 accuracy 测试：配置只影响 kernel 选型，不改变计算逻辑，但不同 tile 分割下浮点累加顺序会变，PR 正文未给数值一致性数据。无新增测试；风险限于该模型+H100+TP=2 组合读到此配置，其他模型/硬件路径不受影响；若 batch 档位与实际负载错配，收益可能为负（PR 未测负向场景）。

## 关联

无（PR 正文未提及同区域其他 PR）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
