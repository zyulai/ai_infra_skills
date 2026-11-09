# PR #9671: Optimized deepseek-v3/r1 model performance on mxfp4 run

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kkHuang-amd |
| 合并 | 2025-09-03 |
| 分类 | Quantization (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/9671 |

## 一句话结论

gfx95 上 DeepSeek-V3/R1 MXFP4 推理把激活量化融合进 RMSNorm、GEMM、RoPE-cat、flatten 等算子并新增 aiter router GEMM,bench_serving 吞吐约 +10%。

## 动机

MXFP4 运行中,激活张量在进入各算子前需独立做 dynamic quant,产生大量额外的量化 kernel 与中间张量:layernorm 输出再量化、GEMM 前量化、attention 输出 flatten 后量化、q/k concat 前后多次转换等。这些独立步骤的访存与 launch 开销在 gfx95 的 WMXFP4 推理路径上占比可观,值得把量化行为并入相邻算子一次性完成。

## 优化点

以 `_use_aiter_gfx95`(SGLANG_USE_AITER + gfx95)为门控,量化被融合进多处:(1) `fused_rms_mxfp4_quant` 把 RMSNorm 与 mxfp4 量化合并,用于 input_layernorm(communicator `prepare_attn`,按 quant_format 触发)与 MLA 的 q_a/kv_a 双 layernorm;(2) quark w4a4 mxfp4 scheme 的 `apply_weights` 支持 tuple 输入 `(x, x_s)`/`(x, x_s, y)` 与 `gemm_afp4wfp4_pre_quant`(量化融进 GEMM 的 atomic 变体),配合 `BumpAllocator` 预分配且零初始化的 router/MoE 输出缓冲,`aiert_dsv3_router_gemm` 用 `gemm_a16w16_atomic` 在 M≤256 时直接写 fp32 累加;(3) MLA 的 `w_kc`/`w_vc` bmm 改 `batched_gemm_afp4wfp4_pre_quant`(mxfp4 权重直乘 fp8/量化激活,免 bf16 上变频与 scale 乘法);(4) `fused_qk_rope_cat` 把 RoPE 与 q/k concat 合一;(5) `fused_flatten_mxfp4_quant` 把 attention 输出 flatten+量化合一;另新增 `quark_post_load_weights` 处理 mxfp4 权重的 w_kc/w_vc 拆分(含静态量化权重的反量化再量化路径)与 `is_gfx95_supported()` 工具函数。全部为 ROCm/gfx95 专属分支,CUDA 路径不动。

## 关键改动

```text
python/sglang/srt/models/deepseek_v2.py: MLA/MoE 前向接入 fused quant gemm/router gemm/fused rope-cat/flatten quant
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: apply_weights 支持 tuple 输入与 pre-quant GEMM
python/sglang/srt/layers/communicator.py: prepare_attn 走 fused_rms_mxfp4_quant
python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py: 封装 aiter fused quant/batched pre-quant 算子
python/sglang/srt/layers/rocm_linear_utils.py: aiter_dsv3_router_gemm 与 zero allocator 尺寸
```

```python
+        if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
+            q, k_nope = fused_rms_mxfp4_quant(
+                q, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon,
+                k_nope, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon,
+            )
```

## 性能声明

- PR 正文:"Test below commands, we got about 10% throughput increased"。
- 条件:服务端 `SGLANG_USE_AITER=1`、`ams/DeepSeek-R1-WMXFP4-Preview`、`--tp 8 --chunked-prefill-size 131072`;客户端 `bench_serving --dataset-name random --random-range-ratio 1 --num-prompt 500 --random-input 3200 --random-output 800 --max-concurrency 128`。
- 改前/改后具体数字以截图形式给出,pack 正文未含文本数字,硬件型号未注明;PR 未注明测试条件中的 GPU 之外细节。
- Accuracy Tests 与 Modifications 章节为空,未提供精度数据。

## 验证与回归风险

- 未新增单元测试(checklist 中 Add unit tests / Update documentation 未勾选);正文无精度验证结果,属于明显缺口。
- 回归面:所有改动都在 `_use_aiter_gfx95`/uint8 权重等条件下分支,非 gfx95 HIP 与 CUDA 路径不受影响;风险集中在 gfx95 上 `w_kc/w_scale_k`、tuple 隐藏状态等新约定与 `o_proj`/attention 输入假定的一致性,以及 `isinstance(hidden_states, tuple)` 新形态在各 attention 后端的传播。

## 关联

- 同区域:AMD gfx95 MXFP4 系列,后继方向与 #37413(fp8 two-pool KV)、#40708(aiter topk_gating)同属 ROCm MoE/quant 优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
