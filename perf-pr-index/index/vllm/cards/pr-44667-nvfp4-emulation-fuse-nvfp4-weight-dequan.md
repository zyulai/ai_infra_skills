# PR #44667: [NVFP4][Emulation] Fuse NVFP4 weight dequantization with compute in triton kernel for w13/w2 MOE MLP linears

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-06-26 |
| 分类 | MoE (tier: major) |
| Label | rocm, ready, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 3 个文件(fused kernel、emulation utils 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/44667 |

## 一句话结论

NVFP4 MoE 权重反量化融进 Triton GEMM,Kimi-K2.6-NVFP4 单 token 提速 10.80x。

## 动机

PR 正文 Purpose:在无原生 NVFP4 硬件支持的平台(如 AMD MI350),NVFP4 MoE 权重需以 BF16 反量化后做 MOE emulation。此前实现(#35737、#40033)在调用 `TritonExperts.apply` 前把 `w13`、`w2` 全部专家的权重张量整体反量化,物化出巨大的 BF16 中间张量,增加显存带宽压力。

## 优化点

新增融合 Triton kernel `fused_moe_nvfp4_emulation_kernel`,对 w13 与 w2 两个 MoE MLP linear 把 NVFP4 权重反量化与 GEMM 计算放在同一 kernel 内:每个 packed uint8 字节(两个 FP4 值)只加载一次,拆出高低 nibble 后经 `_e2m1_inline` 解码、乘 per-block FP8 scale 与 w_global_scale,交织成 [K, N] tile 后直接喂给 `tl.dot`。这样既无需在 HBM 物化完整反量化权重,又只按 `expert_ids` 反量化实际命中的专家(类似 w4a16 模型用的 `fused_moe_kernel_gptq_awq`)。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py: 新增 fused_moe_nvfp4_emulation_kernel,内联反量化 + tl.dot,并按 expert_ids 只处理命中专家
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py: 提供 _e2m1_inline 等 nibble 解码工具
tests/kernels/quantization/test_nvfp4_emulation.py: kernel 数值 + 端到端 MoE forward 对齐参考实现
```

关键机制代码(单个 packed byte 就地解码并送入 tl.dot):

```python
        raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)
        low_nibble = raw_bytes & 0x0F
        high_nibble = (raw_bytes >> 4) & 0x0F
        low_decoded = _e2m1_inline(low_nibble)
        high_decoded = _e2m1_inline(high_nibble)
        low_scaled = low_decoded * b_scale
        high_scaled = high_decoded * b_scale
        b = tl.trans(tl.interleave(low_scaled, high_scaled)).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)
```

## 性能声明

PR 正文 Test Plan 给出 `pytest tests/kernels/quantization/test_nvfp4_emulation.py -k test_nvfp4_moe_correctness` 的 unfused/fused median 对比(Kimi-K2.6-NVFP4,TP=8,top_k=4):tokens=1 时 1.843ms → 0.171ms,10.80x;tokens=2 时 1.855ms → 0.177ms,10.51x;tokens=4 时 1.870ms → 0.185ms,10.13x;tokens=16 时 1.958ms → 0.262ms,7.48x;tokens=64 时 2.244ms → 0.799ms,2.81x;tokens=128/256/512/1024 分别为 2.10x/1.70x/1.70x/1.13x。端到端 eval:`VLLM_ROCM_USE_AITER=1 vllm serve nvidia/Kimi-K2.6-NVFP4 --tensor-parallel-size 8` 在 MI350 上跑 GPQA Diamond zero-shot CoT,exact_match(flexible-extract)0.7172、strict-match 0.0556。

## 验证与回归风险

依赖新增 `tests/kernels/quantization/test_nvfp4_emulation.py`(kernel numerics + 端到端 MoE forward 对齐参考实现)与 MI350 上的 GPQA 端到端 eval。影响路径:无原生 NVFP4 硬件平台(如 AMD MI350)走 NVFP4 MoE emulation 的 w13/w2 linear;正文未列出其他硬件或量化方案的影响,亦未给出前后精度对照表,仅报告 fused 路径的 GPQA 分数。

## 关联

正文引用的前序实现:#35737、#40033(整体反量化方案);借用的同类思路 `fused_moe_kernel_gptq_awq`(w4a16 模型,如 moonshotai/Kimi-K2.6,Instinct MI3xx)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
