# PR #25398: [Perf]  Add H100 fused MoE config

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @skyloevil |
| 合并 | 2025-10-18 |
| 分类 | MoE |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅含 1 个新增 JSON 文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/25398 |

## 一句话结论

为 H100 PCIe 新增 E=64,N=768 FP8 fused MoE Triton 调优配置,吞吐 +1.6–2.9%、延迟约 -2%。

## 动机

Triton fused MoE 内核按 `(E, N, dtype, block_shape)` 与设备名查找 `configs/` 下的分 tile 调优 JSON;H100 PCIe 上 FP8 MoE(Qwen3-30B-A3B 这类 `block_shape=[128,128]` 模型)此前缺少/不匹配的调优条目导致内核用了次优的 BLOCK_SIZE 组合,TPOT/ITL 偏高。用 `benchmark_moe.py --tune` 离线搜参后落盘一份针对该形状的配置即可零代码改善性能。

## 优化点

新增配置文件 `E=64,N=768,device_name=NVIDIA_H100_PCIe,dtype=fp8_w8a8,block_shape=[128,128].json`,覆盖 M=1 到 4096 共 18 档的 Triton 参数(`BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`)。小 M 档(1–128)取 `BLOCK_SIZE_M=16`、`num_warps=4/8` 以贴合 decode/small-batch 的 token 数;M≥256 档升到 `BLOCK_SIZE_M=64` 并按档位调 `GROUP_SIZE_M`(1/16/64)与 `num_stages`(3/4),改善 L2 复用与流水线深度。PR 正文表述为把 H100 PCIe 的 FP8 MoE 配置从 `E=128,N=384` 换成 `E=64,N=768`;diff 实际只包含新增该 JSON 文件,未见删除旧配置。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=64,N=768,device_name=NVIDIA_H100_PCIe,dtype=fp8_w8a8,block_shape=[128,128].json: 新增按 M 档的 Triton 调优参数
```

```text
"M": 1    -> BLOCK_SIZE_M=16, N=128, K=128, num_warps=4, num_stages=4
"M": 1024 -> BLOCK_SIZE_M=64, N=256, K=128, num_warps=8, num_stages=4
"M": 4096 -> BLOCK_SIZE_M=64, N=256, K=128, GROUP_SIZE_M=16, num_warps=8
```

## 性能声明

(模型 Qwen/Qwen3-30B-A3B-FP8,H100 PCIe,`--tp-size 2 --enable-expert-parallel`,fp8_w8a8;服务端压测:`vllm bench serve`,random 数据集 input/output 各 7500 token,并发 64,128 条请求)

| 指标 | Before | After | 变化 |
| --- | --- | --- | --- |
| Output token throughput | 1600.05 tok/s | 1626.21 tok/s | +1.6% |
| Peak output token throughput | 2240.00 tok/s | 2304.00 tok/s | +2.9% |
| Total token throughput | 3211.70 tok/s | 3270.29 tok/s | +1.8% |
| Mean / Median / P99 TPOT | 38.88 / 39.22 / 39.81 ms | 38.05 / 38.56 / 38.85 ms | -2.1% / -1.7% / -2.4% |
| Mean / Median ITL | 38.68 / 36.20 ms | 37.88 / 35.45 ms | -2.1% |

作者结论:该配置为 H100 PCIe 上的 FP8 MoE 模型带来约 1.6–2.9% 吞吐提升与约 2% 延迟下降,mean/median/P99 一致改善。

## 验证与回归风险

验证依赖 `benchmarks/kernels/benchmark_moe.py --tune` 生成与 `vllm bench serve` 端到端复测。风险低:纯配置新增,只影响命中 `E=64,N=768` + H100 PCIe + fp8_w8a8 + `block_shape=[128,128]` 的模型;其他形状/设备不受影响。配置标注 `triton_version: 3.4.0`,Triton 版本差异可能使最优参数偏移。

## 关联

同区域 MoE 调优配置类 PR(#25398 属 configs JSON 添加型优化);pack 素材未列出具体关联 PR 编号,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
