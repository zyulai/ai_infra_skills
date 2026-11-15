# PR #63721: [ROCm][Perf][MoE] Fuse SiLU-and-mul with per-token FP8 quant in TritonExperts and skip remote EP routes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-11-11 |
| 分类 | MoE(tier: major) |
| Label | rocm, ready, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63721 |

## 一句话结论

ROCm `TritonExperts` 的 PTPC-FP8 MoE 把 `silu_and_mul` + per-token `scaled_fp8_quant` 融合为一个 Triton kernel 并跳过 EP 远端路由行:EP8 8,192 tokens 两 kernel 867 → 114 µs(7.6x),Muse Spark serving sweep total throughput +0.8% 到 +5.7%。

## 动机

PTPC-FP8 MoE(如 Muse Spark)下 `TritonExperts` 先跑 `silu_and_mul` 再跑 `scaled_fp8_quant` 两个 kernel。EP 下第一个 GEMM 只写 expert 为 local 的行,但 activation 与 quant 仍对全部行执行,路由到远端 expert 的行被白白激活和量化——第二个 GEMM 与 `moe_sum` 永远不读这些行。#56827(closed)曾对 Marlin ReLU2 处理过同类浪费;本 PR 非 #49828(CUDA C++、dense、无 expert mask)的重复。

## 优化点

- 新 leaf kernel `silu_mul_token_quant`(每 program 处理一行):读与第一个 GEMM 相同的 `expert_map[topk_ids]` mask,跳过路由到远端 expert 的行;id 在 `[0, num_experts)` 之外按 `moe_align_block_size` 的同一规则丢弃,且不越界读 `expert_map`(例如 fused shared-expert slot)。
- 数值对齐:在 SiLU 后与乘积后按 `silu_and_mul` 的位置取整,尺度下限沿用 `dynamic_per_token_scaled_fp8_quant`;bf16 下与 unfused 路径逐 bit 一致,fp16 下 FP8 值最多差一个 FP8 step(`tl.sigmoid` 取倒数而 `silu_and_mul` 用除法;超出 fp16 范围的乘积饱和而非变 inf)。
- 接入门控(见摘录):ROCm、SILU、无 clamp、FP8 W8A8 per-token activation quant、无 block shape、无 static `a2_scale`、无 quantization emulation、无 LoRA;其余配置与其他平台保持既有路径。
- 远端行的 FP8 输出与 scale 留为未初始化(无读者),不产生额外同步。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/silu_mul_token_quant.py(新增): SiLU-and-mul + per-token FP8 quant 融合 Triton kernel,按 expert_map 跳过远端/越界行
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: TritonExperts.apply 新增 fused 分支(ROCm + PTPC-FP8 门控)
tests/kernels/moe/test_silu_mul_token_quant.py(新增): 与 unfused 对齐、越界 id 跳过、fused 分支路由三组测试
```

```python
elif (
    # Only benchmarked on ROCm.
    current_platform.is_rocm()
    and activation == MoEActivation.SILU
    and self.activation_config.clamp_limit is None
    and self.quant_config.use_fp8_w8a8
    and self.per_act_token_quant
    and self.block_shape is None
    and a2_scale is None
    and not self.quantization_emulation
    and lora_context is None
):
```

## 性能声明

- Kernel(单张 MI300X,CUDA graph replay,256 experts,top-8):TP8 8,192 tokens(65,536 x 1,024 行)165 → 91 µs(1.8x);EP8 8,192 tokens(local rows 8,310/65,536,width 8,192)867 → 114 µs(7.6x);1,024 tokens:TP8 27 → 17 µs(1.6x)、EP8 141 → 17 µs(8.2x);256 tokens:TP8 10.4 → 3.9 µs(2.6x)、EP8 31 → 8.3 µs(3.7x);64 tokens:TP8 6.1 → 2.4 µs(2.5x)、EP8 12.6 → 5.0 µs(2.5x)。TP8 整体 1.6–1.8x。
- Serving(amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8,TP8 + EP,8x MI300X,`vllm bench serve` random dataset,3 x concurrency prompts 每点(4 at c1),一点一 run,两次 warmup):1k/8k/16k 输入 x concurrency 1–64 全表 +0.8% 到 +5.7% total throughput;最高点 16k/1k c64 8,749 → 9,247 tok/s(+5.7%);median TTFT 在 8k 与 16k 低 7–9%。
- gfx950(MI355X,TP4 + EP,`--load-format dummy`):1k/1k c64 +0.8%、8k/1k c8 +1.2%、8k/1k c64 +2.2%、16k/1k c16 +1.8%。
- 精度:GSM8K(lm-eval,5-shot,1319 题)ep8 base flexible 0.9484 → 本 kernel + #63722 + #63724 的 ep8_stack 0.9545(stderr ±0.006);该 kernel 单独未跑。

## 验证与回归风险

- `pytest tests/kernels/moe/test_silu_mul_token_quant.py`:kernel 与 `silu_and_mul` + `scaled_fp8_quant` 对比(bf16 与 fp16,1–1,000 tokens,H = 256–4,096,EP1/2/4/8,8–256 experts),检查越界 id 被跳过、`TritonExperts.apply` 以正确的 expert mask 走 fused 分支且输出与 unfused 路径完全一致;三个测试均 ROCm-only(唯一走 fused 路径的平台),MI300X 上 15 passed(`vllm/vllm-openai-rocm:v0.32.0` + 分支 Python tree 与镜像编译 ops)。
- 去掉 kernel 的 lane mask 只挂 `hidden=768` 一例(BLOCK=1024,尾部 256 lane masked)。
- 风险:仅 ROCm 生效;分支内断言 fusion 要求 activation 把 N 减半(`cache2_dim * 2 == N`),不减半的 activation 需要自己的 kernel 或条件;fp16 下不保证逐 bit 一致(允许一个 FP8 step)。

## 关联

- #49828(CUDA C++、dense、无 expert mask,非重复实现);#56827(closed,Marlin ReLU2 的同类浪费)。
- 精度一行叠加 #63722(token-sharded MoE block)与 #63724(bf16x3 router)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
