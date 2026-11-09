# PR #52798: [Quant] Use canonical N-first weight format for CT WNA16 MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @HDCharles |
| 合并 | 2026-09-28 |
| 分类 | MoE \| tier: major |
| Label | ready, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52798 |

## 一句话结论

CT WNA16 MoE 统一为 N-first `[E, N, K_packed]`，转置下移到各 backend，无功能变化。

## 动机

PR 正文 Summary 指出现有实现里 `is_transposed` 是根据所选 backend 设置的，把格式关注点混进了 integration 类 `CompressedTensorsWNA16MoEMethod`，前端与后端各自的 layout 约定纠缠在一起。本 PR 让前端统一坚持 canonical E-N-K 格式，再由各 backend 自行转换到所需格式。

## 优化点

正文列出四点：`CompressedTensorsWNA16MoEMethod` 始终使用 N-first `[E, N, K_packed]` 权重布局，把 `is_transposed` 固定为 `False`，移除 integration 层按 backend 分支的 `is_transposed` 逻辑；把 backend 特定的 N-first → K-first 转置下移到 `convert_to_wna16_moe_kernel_format`，覆盖 Marlin、CPU、XPU、Emulation 后端；修正 RDNA3 子类，在 `process_weights_after_loading` 中显式做 N-first → K-first 转置；简化 `get_weight_shape` 只返回 N-first 形状，删去原先按 Flashinfer / Marlin 分支的 `shape_map`。正文强调 No functional change for any backend —— 同样的转置仍会发生，只是位置移到 pipeline 的不同点。

## 关键改动

```text
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py: is_transposed 固定 False，get_weight_shape 只返回 N-first 形状，转置下移到 convert_to_wna16_moe_kernel_format
vllm/model_executor/layers/fused_moe/oracle/int_wna16.py: oracle 侧按 canonical N-first 输入处理，RDNA3 显式转置、CPU 保留 zero points
vllm/model_executor/layers/fused_moe/routed_experts.py、vllm/model_executor/layers/quantization/auto_gptq.py: 相关接线改动
tests/kernels/quantization/test_int4_emulation_moe.py: 新增 _canonicalize_gptq_moe_inputs 把 GPTQ K-first 输入转成 canonical N-first
tests/quantization/test_moe_wna16.py: 更新 N-first pass-through 验证
```

```diff
-        self.is_transposed = self.wna16_backend not in (
-            WNA16MoEBackend.FLASHINFER_TRTLLM,
-            WNA16MoEBackend.HUMMING,
-        )
+        # CT checkpoints always load in N-first format [E, N, K_packed].
+        # Backend-specific transposition is handled in
+        # convert_to_wna16_moe_kernel_format.
+        self.is_transposed = False
```

## 性能声明

PR 正文未给出定量性能数据（纯权重格式重构，正文明确 "No functional change for any backend — same transposes happen, just at a different point in the pipeline"），也未给出测试条件，模型、硬件、精度与 workload 均 PR 未注明测试条件。

## 验证与回归风险

Test plan 三项均勾选：`test_compressed_tensors_nfirst_weights_pass_through_for_triton`（更新为验证 N-first pass-through）、All existing `test_moe_wna16.py` tests pass、No functional change for any backend。diff 含 `tests/kernels/quantization/test_int4_emulation_moe.py`（新增 `_canonicalize_gptq_moe_inputs`，把 GPTQ K-first 张量 `transpose(1, 2).contiguous()` 成 canonical N-first 后再喂给 oracle）与 `tests/quantization/test_moe_wna16.py`。回归风险：改动触及所有 CT WNA16 MoE backend（Marlin、CPU、XPU、Emulation、RDNA3）的权重布局与转置位置，以及 `auto_gptq.py`、`routed_experts.py` 等接线；`int_wna16.py` 中 zero points 需求也由 CPU/RDNA3 收窄为仅 CPU，正文称无功能变化，PR 未注明更多受影响路径或硬件。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
