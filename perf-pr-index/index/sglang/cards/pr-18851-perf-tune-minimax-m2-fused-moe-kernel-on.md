# PR #18851: [Perf] Tune MiniMax M2 fused moe kernel on H100 GPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zhendonghua |
| 合并 | 2026-02-15 |
| 分类 | MoE (major) |
| Label | ready-to-merge |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 为新增 1 个 tuned config JSON(146 行) |
| 链接 | https://github.com/sgl-project/sglang/pull/18851 |

## 一句话结论

为 MiniMax-M2 在 H100 上新增 fused_moe_triton 调优配置,decode 输出吞吐 960→1008 token/s,约 5% 提升。

## 动机

MiniMax-M2 此前没有针对 fused_moe_kernel 与 H100 GPU 的调优配置,fused MoE 走默认参数,decode 吞吐次优。MoE 模型的 fused_moe_kernel 是 decode 阶段的热点,块大小、warps、stages 等 tile 参数未按模型形状(E=256、N=384)与硬件调过时会明显偏离最优。

## 优化点

用官方调优脚本 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`(`--model MiniMaxAI/MiniMax-M2 --tp-size 4 --dtype fp8_w8a8 --disable-shared-experts-fusion --tune`)离线扫描,新增 `triton_3_5_1/E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` 配置。覆盖 batch size 1-4096 共 18 个档位,按 M 值选择 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages:小 batch(1-96)档以 BLOCK_SIZE_M=16、BLOCK_SIZE_N=64、num_stages 3-5 为主,降低 tile 粒度提高占用率;大 batch(1024+)档切换到 BLOCK_SIZE_M=64、GROUP_SIZE_M=16;4096 档单独用 num_stages=4。机制上即为该模型形状在 H100 上重新平衡 SM 占用与流水线深度,无运行时代码改动。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json: 新增文件,18 个 batch 档位的 Triton tile 配置
```

典型档位(小 batch decode 区间与最大档):

```json
"1":  { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,  "num_warps": 4, "num_stages": 5 },
"64": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,  "num_warps": 4, "num_stages": 3 },
"4096": { "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 4 }
```

## 性能声明

测试条件:MiniMaxAI/MiniMax-M2,TP4,H100 80GB HBM3,fp8_w8a8 + block_shape [128,128],triton_3_5_1 配置目录;服务端 `--mem-fraction-static 0.85`。

- E2E decode:bench_serving random 数据集,input 1024 / output 1024,range-ratio 0.98,80 prompts、max-concurrency 16。tuned 后 output throughput **1008 token/s**,未 tuned 为 960 token/s,约 **5% speedup**。
- Kernel 级 profile:bench_serving(input 1200 / output 20,concurrency 1)profiling 显示 tuned fused_moe_kernel 约 **12 us**,未 tuned 约 **15.5 us**。

PR 的 Accuracy Tests 部分为空(未提供准确率数据)。

## 验证与回归风险

无新增测试(checklist 中 unit tests 未勾选);验证手段为上述 bench_serving 对比与 torch profiler 截图。风险面:配置按 `device_name=NVIDIA_H100_80GB_HBM3` + E=256/N=384 + dtype/block_shape 精确匹配,只影响 H100 上该模型形状的 fused MoE kernel 选择,且放在 `triton_3_5_1` 版本目录下、与 Triton 版本绑定;调优时使用 `--disable-shared-experts-fusion`,若部署开启 shared experts fusion 则形状可能不同。其他卡型与其他模型不受影响。

## 关联

同区域:H20 tuned config PR #5716(同类 fused_moe_triton 配置调优);MiniMax-M2 模型接入相关 PR(pack 未列编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
