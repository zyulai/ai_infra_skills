# PR #23939: Tuned H100/H200 triton fp8 block configs for fused_qkv_a_proj

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-08-29 |
| 分类 | Quantization(major) |
| Label | performance |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 内容可用(2 个 config JSON + 2 个 benchmark 脚本) |
| 链接 | https://github.com/vllm-project/vllm/pull/23939 |

## 一句话结论

为 DeepSeek-V3 fused_qkv_a_proj 的 N=2112,K=7168 形状补上 H100/H200 的 W8A8 Block FP8 Triton tuned config,H100 上 batch=1 的 GEMM 从 0.53 提到 1.30 TFLOP/s(约 2.5 倍)。

## 动机

该形状此前没有对应 config JSON,H100 上运行时打出警告 "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!"。默认 config 对小 batch(decode 场景的 batch=1/16/17)尤其吃亏:torch-bf16 GEMM 有 3.1 TFLOP/s,FP8 Triton kernel 反而只有 0.53 TFLOP/s,量化路径比非量化慢一个量级。

## 优化点

新增 `N=2112,K=7168` 的 H100_80GB_HBM3 config JSON(18 个 batch 档位,覆盖 M=1 到 4096,每档给出 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages),小 batch 档位选用 BLOCK_SIZE_M=16、BLOCK_SIZE_N=32 的窄 tile 贴合 decode 的窄 GEMM;同时把已有的 H200 同名 config 从只有 "2048" 一档补全为同样的全档位表。并把 (2112, 7168) 加入 `benchmarks/kernels/bench_block_fp8_gemm.py` 与 `benchmark_w8a8_block_fp8.py` 的 shape 列表,使该形状进入常规 benchmark 覆盖。不改变 kernel 代码,纯 tuning 配置补齐。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 tuned config
vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json: 从单档补全为全档位
benchmarks/kernels/bench_block_fp8_gemm.py、benchmark_w8a8_block_fp8.py: 增加 (2112, 7168) 形状
```

```json
"1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 5
}
```

## 性能声明

PR 正文(H100 80GB HBM3,DeepSeek-V3 形状 N=2112 K=7168,block_size=(128,128),w8a8-block-fp8-triton 列,TFLOP/s):
- batch 1:0.527570 → 1.303974(约 2.5 倍)
- batch 16:9.050292 → 20.520697;batch 17:9.802354 → 21.695363
- batch 512:215.180092 → 307.285148
- batch 4096:305.138408 → 392.939256;batch 16384:354.501542 → 403.475538

同表 torch-bf16 对照(batch 512 约 561.8、batch 4096 约 728.7)。硬件仅注明 H100;H200 config 有改动但 PR 未附 H200 数字。

## 验证与回归风险

无新增单测;验证即 benchmark 脚本的 before/after TFLOP/s 对比表。回归风险低:config 按 (N,K,device_name,dtype,block_shape) 精确匹配加载,其他形状/设备不受影响;数值为离线 tuning 结果,Triton 版本变化后最优配置可能偏移。

## 关联

同区域为 vLLM 的 W8A8 Block FP8 Triton GEMM configs 目录(其他 N,K/device 组合的 JSON);素材未给出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
