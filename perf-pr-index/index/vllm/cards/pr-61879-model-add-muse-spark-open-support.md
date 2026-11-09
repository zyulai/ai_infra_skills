# PR #61879: [Model] Add Muse Spark Open support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-10-26 |
| 分类 | Other（tier: major） |
| Label | performance, new-model, rocm, frontend, speculative-decoding, ci/build, tool-calling, nvidia, quantization, mrv2, dflash, rust, kimi, k3 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "42 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/61879 |

## 一句话结论

新增 Muse Spark Open 模型支持（含 NVIDIA/ROCm MoE、norm、qk_norm 测试与 benchmark）；PR 未给出定量性能数据。

## 动机

PR 正文仅一句："Depend on https://github.com/flashinfer-ai/flashinfer/pull/6544"，未提供改动目的、瓶颈或性能说明，按规则写「PR 未注明」。从 title 与 Label 仅可知这是 Muse Spark Open 的新模型支持改动。

## 优化点

PR 正文未说明算法/内核/调度层面的优化机制，按规则写「PR 未注明」。从可见 diff 看主要是该模型相关测试与 benchmark 的配套改动：`benchmarks/kernels/benchmark_muse_spark_qknorm_rope_kv_insert.py` 增加 NoPE 基线（`unfused_nope`，调用 CuTe-DSL `qk_norm`）与 `HAS_CUTEDSL` 探测，并把 `modes` 从 `["rope"]` 扩为 `["rope", "nope"]`；`tests/kernels/core/test_muse_spark_norms.py` 从 AMD 的 `model_ops` 切到 `vllm.models.muse_spark.nvidia.model`，新增 `requires_b300_dual_norm`（SM103 / CuTe DSL 的 FlashInfer dual norm）跳过条件与 `test_flashinfer_residual_boundary_matches_triton` 用例，并在 Triton 边界测试中 monkeypatch 掉 `gated_residual_next_rmsnorm`；另有多个 `tests/kernels/moe/test_muse_spark_*.py` 相关测试。pack 因体积丢弃 42 个文件，模型实现主体未在 diff 中展示。

## 关键改动

```text
benchmarks/kernels/benchmark_muse_spark_qknorm_rope_kv_insert.py: 加入 NoPE / CuTe-DSL qk_norm 基线（modes 增加 "nope"）
tests/kernels/core/test_muse_spark_norms.py: 切换到 nvidia.model_ops，新增 B300 dual norm（SM103）测试与 monkeypatch
tests/kernels/moe/test_muse_spark_*.py: MoE finalize / router / deployment / TRTLLM / ROCm bf16 / Hopper marlin 等测试
```

```diff
+try:
+    from vllm.models.muse_spark.nvidia.ops.cute_dsl.qk_norm import (
+        qk_norm,
+        supports_qk_norm,
+    )
+
+    HAS_CUTEDSL = supports_qk_norm(torch.bfloat16, HEAD_DIM)
+except ImportError:
+    HAS_CUTEDSL = False
```

## 性能声明

PR 正文未给出定量性能数据（perf claims 段标注 "(none detected by regex)"，正文仅一句依赖说明）；PR 未注明测试条件。注意 diff 中存在一个 benchmark 脚本（`benchmark_muse_spark_qknorm_rope_kv_insert.py`）的改动，但正文与 pack 均未给出其运行结果数值。

## 验证与回归风险

验证：可见 diff 新增/扩展 `tests/kernels/core/test_muse_spark_norms.py`（Triton 边界 255/256 rows、B300 dual norm 与 Triton 一致性）、多个 `tests/kernels/moe/test_muse_spark_*.py`（muse_spark_utils、hopper_marlin_moe、moe_deployment、moe_finalize、rocm_bf16_moe、router、trtllm_moe）以及上述 benchmark 脚本。PR 正文未注明测试结果与回归风险；pack 丢弃 42 个文件，无法确认其余改动内容。

## 关联

依赖 flashinfer-ai/flashinfer 的 PR #6544（正文 "Depend on https://github.com/flashinfer-ai/flashinfer/pull/6544"），属于外部仓库依赖而非本仓库 PR 编号；未给出本仓库其他 PR 编号或前序/后续线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
