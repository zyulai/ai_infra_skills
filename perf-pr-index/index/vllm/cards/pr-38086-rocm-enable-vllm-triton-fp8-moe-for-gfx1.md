# PR #38086: [ROCm] Enable VLLM triton FP8 moe for gfx1201, tuned for Qwen3-30B-A3B-FP8 tp=2 and Qwen/Qwen3.5-35B-A3B-FP8 tp=2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vllmellm |
| 合并 | 2026-04-02 |
| 分类 | MoE(major) |
| Label | performance, rocm, ready, qwen |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/38086 |

## 一句话结论

在 RDNA4(gfx12xx)上启用 vLLM Triton FP8 MoE,并为 Qwen3-30B/35B-A3B-FP8 补上 R9700 调优配置,TPOT 降低约 8%。

## 动机

FP8 MoE 模型此前无法在 RDNA4(gfx12xx)GPU 上启动:`_supports_quant_scheme` 只认 gfx9/CUDA(8,9)/XPU,报 `NotImplementedError: No FP8 MoE backend supports the deployment configuration`(issue #36105)。同时 gfx1201(AMD Radeon R9700)没有任何 fp8_w8a8 block-quant 的 MoE kernel 调优配置,即使能跑也只能用默认参数。

## 优化点

三层改动:(1) 平台识别——`vllm/platforms/rocm.py` 新增 `on_gfx12x()` 判定与 device_id `0x7551 → AMD_Radeon_R9700` 映射,顺带修正 `get_device_name` 中 device_id 变量遮蔽;(2) 量化方案判定——`fused_moe.py` 的 `_supports_quant_scheme` 把 `is_rocm_on_gfx12x` 纳入 `device_supports_fp8`,使 Triton FP8 MoE 后端在 gfx12xx 上可用;(3) 调优——新增两份配置 `E=128,N=512` 与 `E=64,N=768`(device AMD_Radeon_R9700,dtype fp8_w8a8,block_shape=[128,128]),按 M=1..4096 分档给出 BLOCK 尺寸/warps/stages 及 ROCm 特有的 `waves_per_eu`,triton_version 标注 3.6.0。

## 关键改动

```text
vllm/platforms/rocm.py: on_gfx12x() 判定 + R9700 device_id 映射
vllm/model_executor/layers/fused_moe/fused_moe.py: gfx12x 计入 device_supports_fp8
vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=AMD_Radeon_R9700,...json: Qwen3-30B 调优表
vllm/model_executor/layers/fused_moe/configs/E=64,N=768,device_name=AMD_Radeon_R9700,...json: Qwen3.5-35B 调优表
```

关键片段(fused_moe.py):

```python
device_supports_fp8 = (
    is_rocm_on_gfx9
    or is_rocm_on_gfx12x
    or (p.is_cuda() and p.has_device_capability((8, 9)))
    or p.is_xpu()
)
```

## 性能声明

测试条件:2 块 Radeon PRO 9700,`-tp 2 --enable-expert-parallel --gpu-memory-utilization 0.95 --max-model-len 65536`(Qwen3-30B 另设 `VLLM_ROCM_USE_AITER=0`)。Tuned vs Untuned:
- Qwen3-30B-A3B-Instruct-2507-FP8:TPOT 平均 53.51 → 48.95 ms;E2E 平均 121743.70 → 113552.15 ms;TTFT 平均 24749.45 → 24955.37 ms(其中 ISL16384-OSL2048 一档 TTFT 126817.12 → 131106.68,变差)。
- Qwen3.5-35B-A3B-FP8:TTFT 平均 9543.10 → 7489.73 ms;TPOT 平均 54.75 → 50.73 ms;E2E 平均 107317.17 → 98432.20 ms。
GSM8K exact_match:Qwen3-30B strict 83.24% → 83.40%、flexible 85.82% → 86.43%;Qwen3.5-35B strict 86.50% → 87.11%、flexible 87.49% → 88.02%。

## 验证与回归风险

验证为上述 benchmark 与 GSM8K 精度对比。风险面:平台判定放宽只影响 gfx12xx,但 `_supports_quant_scheme` 是共享函数,需确认 gfx12 之外的 ROCm 架构(如 gfx11)不被误开 FP8 路径;配置文件按键精确匹配,E/N 不符则不受影响;`get_device_name` 的变量改名属行为中性修正。PR 未注明新增自动化测试。

## 关联

解决 issue #36105;同区域:PR #31407、PR #46642 等 fused MoE configs 调优类 PR,以及 PR #58569 等 ROCm 路径优化。PR 未注明其他前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
