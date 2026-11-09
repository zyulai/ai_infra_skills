# PR #54894: [ROCm][DSV4][Perf] Use FP8 WO_A output projection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LiuYinfeng01 |
| 合并 | 2026-09-20 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, deepseek, verified, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/54894 |

## 一句话结论

ROCm gfx950 上 DSV4 改用原生 FP8 `wo_a` 输出投影，100K prefill TTFT -7.35%、输入吞吐 +7.93%。

## 动机

正文 Summary 指出，ROCm DeepSeek V4 在 gfx950 上的 `wo_a` 输出投影走 BF16 grouped einsum，而 checkpoint 本就带原生 FP8 `wo_a` 权重。此 PR 用 AITER 的 `inverse_rope_group_quant`（把 inverse RoPE 与 per-token E8M0 group quant 融合）和 `batched_gemm_a8w8_mxscale`（吃量化激活与 checkpoint 原生 FP8 `wo_a` 权重）替换它，去掉 BF16 grouped einsum，让 `wo_a` 保持 checkpoint 的 FP8 形式；不支持的权重、scale、设备与旧版 AITER 保留 BF16 fallback。

## 优化点

(1) 复用既有 `VLLM_ROCM_USE_AITER_FP8BMM` gate 并把路径限制在 gfx950（`_ON_GFX950`）；依赖 AITER >= 0.1.20（MX-scale batched GEMM，见 #52826），capability probe 只记录一次并在 op 不可用时回退 BF16。(2) `wo_a` block scale 接受原生 `float8_e8m0fnu`/`uint8`，或把正的 2 的幂 FP32 scale 无损转为带 bias 127 的 E8M0 指数字节（`#0xFF` 为 NaN、越界返回 None），启用前校验 group-128 权重与 scale 布局。(3) 权重加载后一次性缓存连续的 BF16 cos/sin cache views；保留 `rocm_inv_rope_einsum` 作为 fallback。(4) 新增 `_wo_a_block_scale_to_e8m0` 与 `_prepare_fp8_wo_a`，并新增对应单测。

## 关键改动

```text
vllm/models/deepseek_v4/amd/rocm.py: _wo_a_block_scale_to_e8m0、_prepare_fp8_wo_a、gfx950+FP8BMM gate、FP8 权重/scale/cos/sin 缓存，保留 BF16 fallback
tests/models/test_deepseek_v4_rocm_wo_a.py: 新增，覆盖 E8M0 转换、已编码 scale 保留与非法 scale 拒绝
```

关键片段（E8M0 scale 归一化）：

```diff
+    if scale.dtype == torch.float8_e8m0fnu:
+        # Reinterpret the native E8M0 storage. ``to(uint8)`` would perform a
+        # numeric conversion instead of preserving the encoded exponent byte.
+        return scale.view(torch.uint8).contiguous()
+    if scale.dtype == torch.uint8:
+        # The checkpoint loader already exposed the E8M0 wire representation.
+        return scale.contiguous()
+    if not scale.dtype.is_floating_point:
+        return None
+
+    scale_f32 = scale.detach().float()
+    if not bool(torch.isfinite(scale_f32).all()) or bool((scale_f32 <= 0).any()):
+        return None
```

## 性能声明

主结果（8 x MI355X gfx950、TP1/PP8、单条 100,000-token prefill、8K prefill chunks、prefix caching disabled、3 runs）：OPUS sparse MLA + tuned GEMMs 的 TTFT runs 2470.2 / 2457.9 / 2459.0 ms、median 2459.0 ms、Input throughput 40,667.0 tok/s；`+ FP8 wo_a` 为 2288.8 / 2276.1 / 2278.4 ms、median 2278.4 ms、43,891.0 tok/s；Delta **TTFT -7.35%**、**input throughput +7.93%**。TP8/PP1 与 TP1/PP8 的 prefill 并发扫描（8 x MI355X、固定 8,192-token prompts、1 output token、无 prefix cache、每并发 10 请求、每臂 1,110 请求全成功）：TP8/PP1 增益 +0.78%~+1.90%，TP1/PP8 增益 +3.68%~+6.73%。旧 `DeepSeek-V4-Pro` checkpoint 上另测（8 x MI355X、FP8 KV、A/B gate `VLLM_ROCM_USE_AITER_FP8BMM=0/1`）：TP8/PP1 mean +1.29%、TP1/PP8 mean +6.09%；TP8/PP1 decode（8K/1K）mean TPOT -1.85%~-3.39%、output throughput +1.89%~+3.23%；100K prefill TTFT -2.89%、input throughput +2.98%；GSM8K full 1319 两臂均 94.7688%、零 invalid，output tok/s +1.11%。正文明确 0813 与旧 checkpoint 两组「not interchangeable」，分开并列。

## 验证与回归风险

测试 revisions：vLLM PR commit `a9cf3bc3498eb71bb2d6f5d44c4b0049b62feb7a`、AITER commit `a6d2b564fd671724a3720b8edf70e8d674e4d694`、FlyDSL 0.3.1；common settings 为 TP8/PP1/DCP1/PCP1/EP1、FP8 KV cache、AITER MoE、Triton sparse prefill、8K max batched tokens、无 prefix cache，enabled-arm 日志加载 `module_inverse_rope_group_quant` 与 `opus_bmm_a8w8_mxscale`、disabled-arm 均无。`git diff --check` 通过；两个改动文件的 pre-commit 通过；`tests/models/test_deepseek_v4_rocm_wo_a.py` 7 passed；TP1/PP8 100K prefill 3/3 完成；GSM8K-1319 零 invalid。GSM8K 正确性：100-question gate 93/100、full 1319 1256/1319=95.22%、clean fixed-stack reference 1249/1319=94.69%；TP8/PP1 GSM8K 上 BF16 `wo_a` 93.9348%（1239/1319，0 invalid）vs FP8 `wo_a` 94.9204%（1252/1319，1 invalid）。风险：仅 gfx950 且需 AITER >= 0.1.20（依赖 #52826），旧 AITER 回退 BF16；A/B gate 为 `VLLM_ROCM_USE_AITER_FP8BMM=0/1`；两组 checkpoint 数据不可互换；FP8 路径改变数值行为（测量为 accuracy-neutral）。

## 关联

#52826（AITER >= 0.1.20 依赖，MX-scale batched GEMM）；#45103（fused inverse RoPE + 缓存 BF16 `wo_a`，本 PR 保留为 fallback 并在受支持的 gfx950 配置改用原生 FP8 权重）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
