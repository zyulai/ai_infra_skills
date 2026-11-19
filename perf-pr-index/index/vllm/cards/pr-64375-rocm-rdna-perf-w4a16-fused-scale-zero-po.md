# PR #64375: [ROCm][RDNA][Perf] W4A16: fused scale/zero-point carrier for Triton prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @stoivonen-amd |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, rocm, ready |
| Diff | pack 标注 +None/-None, None files；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/64375 |

## 一句话结论

gfx1151 (Strix Halo) 的非对称 fp16 W4A16 层在加载时把每 (n, group) 的 scale 与 zero-point 打包成一个 int32 carrier，prefill 反量化由 scale load + packed zp gather + sub-mul 变为一次 load + 单条 `v_pk_fma_f16`，batch-1 prefill 端到端最高快 18.4%。

## 动机

#60644 的 packed fp16 Triton prefill 反量化路径上，非对称层每个 (n, group) 要做一次 scale load、一次 packed zero-point gather 和先减后乘；对称层只需一次 scale load。把 scale 与 `bias_eff = -zp * scale` 预先折叠进一个 int32（低 16 位 scale、高 16 位 bias_eff），magic-constant unpack 后的反量化就变成一条 FMA。代价是少量权重内存（g128 为 int4 权重字节的 6.25%、g64 为 12.5%）。由此非对称层可从 M > 192（N ≥ 4K 时 M > 128）就走 packed 路径，而不是原来的 M > 256。构建于 #60644 之上；#61193 也在加速 gfx1151 prefill（M ≥ 4096 时反量化成 dense matmul），互补但改同一文件。

## 优化点

- **Carrier 布局**：`[N, K//G]` int32，低 16 位为 fp16 scale，高 16 位为 fp16 `bias_eff = -zp * scale`；反量化 `(b_raw - 1024) * scale + bias_eff`。输出与 zero-point 路径非 bit-identical（bias_eff 舍入到 fp16 并融入 FMA）。
- **构建条件**：仅非对称层、fp16 scales、g ≥ 64、gfx1151（packed dequant 唯一运行地）。以 `Parameter` 存储，layerwise `reload_weights` 重算并拷回原存储（non-persistent buffer 会保留陈旧值，新 reload 测试会抓出）。scalar dequant（fp16 小 M、全部 bf16）上 carrier 平均慢约 1%（部分 cell 慢 10–20%，VALU-issue-bound），不启用。
- **新 cutoff 逻辑**（`_select_skinny_gfx1151_config`）：bf16 不走 packed ladder；有 carrier 时 `M > 192 or (M > 128 and N >= 4K)`；有 zp（无 carrier）时 `M > 256`；对称 `M > 128`。
- **不改动**：对称层、bf16、g32（与 main bit-identical）、skinny HIP decode 路径（`wvSplitK_int4_g` 仍读原 scales/zp，carrier 是额外拷贝；decode 改读 carrier 留作后续）、gfx1151 以外的所有 GPU。

## 关键改动

```python
# vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py
+def pack_scale_zp_carrier(scales, zp):
+    """低 16 位 scale、高 16 位 bias_eff = -zp * scale; 返回 [N, K//G] int32 carrier."""
+    zp_raw = ((zp[:, None, :] >> shifts[None, :, None]) & 0xF).reshape(n_rows, num_groups)
+    bias_eff = -(zp_raw.to(torch.float32) * scales.to(torch.float32))
+    hi = bias_eff.to(torch.float16).view(torch.int16).to(torch.int32)
+    lo = scales.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
+    return ((hi << 16) | lo).contiguous()
# kernel 内:
+        if HAS_SCALE_ZP_CARRIER:
+            scale_zp_carrier = tl.load(scale_zp_carrier_ptr + offs_n * num_groups + group_idx, ...)
+            scales = (scale_zp_carrier & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
+            bias_eff = ((scale_zp_carrier >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
...
+            if HAS_SCALE_ZP_CARRIER:
+                # bias_eff = -zp * scale, so the affine is one v_pk_fma_f16.
+                b_fp = (b_raw - c1024) * scales[:, None] + bias_eff[:, None]
# dispatch:
-    min_m = 256 if has_zp else 128
-    if dtype == torch.float16 and min_m < M:
+    elif has_scale_zp_carrier:
+        use_packed_ladder = M > 192 or (M > 128 and N >= 4 * K)
# process_weights_after_loading: 满足条件时 replace_parameter(layer, "_hybrid_w_scale_zp_carrier", pack_scale_zp_carrier(w_s, w_zp))
benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py: 新 provider "hybrid-w4a16-scale-zp"
tests/kernels/quantization/test_rdna_hybrid_w4a16.py: zp_mode 参数化 (sym/zp/scale_zp_carrier) +
  test_rdna_hybrid_w4a16_reload_rebuilds_scale_zp_carrier（layerwise/direct reload, data_ptr 不变、值重建）
```

## 性能声明

（环境：gfx1151，Radeon 8060S，20 CUs，torch 2.13.0+rocm10.2.0a20260923，Triton 3.8.0。e2e 对比 main@4b47db922b，kernel 对比 main@0fddee1f58；两臂同 build 交错运行、每 rep 交替先后。）

- **Batch-1 prefill（`vllm bench latency`，8 runs/臂取中位）**：Qwen3-8B（asymmetric g128）224 tokens 166.4 → 135.9 ms（−18.4%）、256 tokens 168.0 → 138.9（−17.3%）；Qwen3-1.7B（asymmetric g64）224 41.5 → 36.4（−12.1%）、256 42.3 → 37.0（−12.4%）；128 tokens 不变、320 tokens 起在 ±1.8% 内。
- **Batch 128/256 无回退**：Qwen3-8B 快 3–11%（如 batch 256/128 in/128 out 50.22 → 44.83 s，−10.7%）、Qwen3-1.7B 快 1–5%。
- **Kernel**（`triton.testing.do_bench`，flush cache，3 reps 中位，12 个 Qwen3 形状 geomean，PR/main 时间比）：M 193–256 快 15–18%（0.82–0.84），M 129–192 快 4–5%（0.95–0.96），M 257–4096 快 1–3%（0.97–0.99）；M=8192 按各模型四层求和 Qwen3-8B 快 12%、Qwen3-1.7B 快 5%（分层比：qkv 0.91/0.96、o 0.90/0.99、gate_up 0.91/0.95、down 0.81/0.91）。唯一回退是 Qwen3-4B down_proj（N=2560, K=9728）M 193–256 的 1.16–1.24×，四层求和仍 0.85–0.89×（g128）。
- **内存**：int4 权重字节的 8/G（g128 6.25%、g64 12.5%）；实测 KV cache 反而小 0.16–0.20%。
- **精度**：GSM8K 5-shot 500 题，#60644 head → 本改动：Qwen3-8B 0.884 → 0.882、Qwen3-1.7B 0.654 → 0.652（各一题之差）；对 fp32 反量化-再矩阵乘参考的相对误差 3.98e-4（carrier）vs 2.86e-4（无 carrier）；对称输出与 main bit-identical（168/168 cells）；g32 无 carrier 时 252/252 cells bit-identical、时间比 0.99–1.00。

## 验证与回归风险

`pytest tests/kernels/quantization/test_rdna_hybrid_w4a16.py tests/kernels/quantization/test_w4a16_kernel_selection.py`：gfx1151 上 338 passed、8 skipped（bf16 carrier 用例）。新增/更新测试：prefill GEMM 对 fp32 oracle 的 sym/zp/carrier 三模式、`reload_weights` 在原存储重建 carrier（layerwise 与 direct 两种 reload 模式，断言 `data_ptr` 不变且值等于新权重重打包结果）。CI 覆盖缺口：AMD quantization-kernels 步在 gfx942 上跑这两个文件，carrier 测试在那里 skip（GEMM 测试需 gfx11/gfx12、reload 测试需 gfx1151），故 carrier 路径在 gfx1151 上于 CI 外验证。回归风险：仅 gfx1151 + 非对称 fp16 + g≥64 生效；输出与 zero-point 路径非 bit-identical（bias_eff fp16 舍入）；Qwen3-4B down_proj 形状在 M 193–256 有 16–24% 单层回退；与 #52637、#61193 在同文件上存在合并交互；#60710 的 dense weight cache 启用时 carrier 只在 M 129–256 有意义；PR checklist 中 Design Fit 等三项未勾选。

## 关联

- 构建于 #60644（packed fp16 Triton prefill dequant）；解码 kernel `wvSplitK_int4_g` 来自 #57199。
- 相关 open PR：#52637（同文件 `process_weights_after_loading`）、#61193（同文件 prefill dispatch，M ≥ 4096 dense matmul）、#60710（opt-in dense weight cache）。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
