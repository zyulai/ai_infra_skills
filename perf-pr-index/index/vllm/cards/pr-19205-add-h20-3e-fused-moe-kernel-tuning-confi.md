# PR #19205: Add H20-3e fused MoE kernel tuning configs for DeepSeek-R1/V3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Xu-Wenqing |
| 合并 | 2025-06-05 |
| 分类 | MoE(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用(单文件新增) |
| 链接 | https://github.com/vllm-project/vllm/pull/19205 |

## 一句话结论

新增 H20-3e 上 DeepSeek-R1/V3 的 fused MoE fp8_w8a8 调参配置,端到端输出吞吐 +16%。

## 动机

H20-3e 是 141GB 显存的 H20 变体,vLLM 配置库里没有对应 device_name 的 fused MoE tuning 配置;在 H20-3e 上部署 DeepSeek-R1/V3(TP8,fp8_w8a8,block_shape=[128,128])时 fused_moe kernel 落到默认配置,suboptimal。

## 优化点

纯配置新增:用 benchmark_moe.py --tune 在 H20-3e 上为 DeepSeek-R1-0528 的 MoE 形状(E=256,N=256,fp8_w8a8,block_shape=[128,128])逐 batch size 调参,产出 E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json。配置形态:batch 1-512 段 BLOCK_SIZE_M=16、BLOCK_SIZE_N=128 为主,BLOCK_SIZE_K 在 128/256 间选择;batch 1024 以上切到 BLOCK_SIZE_M=64;num_warps=4、num_stages=3/4 为主。不改任何 kernel 代码。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json: 新增 18 个 batch size 档位的 Triton 配置
```

```json
"1": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
       "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3 },
"1024": { "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
          "GROUP_SIZE_M": 64, "num_warps": 4, "num_stages": 3 }
```

## 性能声明

DeepSeek-R1-0528,TP8,H20-3e,sglang.bench_serving random 1024 输入/512 输出,max-concurrency 8,200 prompts(开/关 tuning 配置对比):

| 指标 | 无 tuning | 有 tuning |
| --- | --- | --- |
| 输出 token 吞吐 | 194.15 tok/s | 225.04 tok/s |
| 总 token 吞吐 | 567.31 tok/s | 657.58 tok/s |
| Mean ITL | 39.88 ms | 34.17 ms |
| Median ITL | 37.86 ms | 32.04 ms |
| Mean E2E Latency | 10761.65 ms | 9255.99 ms |
| Mean TTFT | 162.99 ms | 157.92 ms |

## 验证与回归风险

验证即上述端到端 serving benchmark;无 kernel 单元测试(纯配置)。风险很低:配置按 device_name 命中,只影响 NVIDIA_H20-3e 设备上的该 MoE 形状;其他设备/形状不受影响。

## 关联

同分类 MoE 卡:#14068(block quant tuning 脚本支持,config 文件名含 block_shape 的约定来源)、#39904(gpt-oss H100 fused_moe 配置)、#23748(block-FP8 GEMM H100/H200 配置)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
