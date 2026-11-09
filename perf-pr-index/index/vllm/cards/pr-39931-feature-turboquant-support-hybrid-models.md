# PR #39931: [Feature] TurboQuant: support hybrid models and uniform quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JartX |
| 合并 | 2026-05-05 |
| 分类 | Quantization \| tier: major |
| Label | rocm, ready, v1, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/39931 |

## 一句话结论

修复 TurboQuant 在 Qwen3.5/Qwen3-Next 等 hybrid 模型上遇 Mamba 层即抛 NotImplementedError 的启动失败。

## 动机

PR 正文：本 PR 修复 TurboQuant 在 hybrid 模型（如 Qwen3.5、Qwen3-Next 及类似架构）上的启动问题。此前 TurboQuant 一遇到 Mamba 层就会抛出 `NotImplementedError`；改动后 hybrid 模型被正确处理，TurboQuant 只施加于 `full_attention` 层。

## 优化点

除 hybrid 支持外，正文列出三项附带修复：一是 page-size planner 不匹配——hybrid page-size planner 用标准公式估算 attention page，与 TurboQuant 的 packed K|V 布局不符，可能导致每个 TurboQuant attention 层在 page merger 触发断言，现改用 TurboQuant 专属布局（`TQFullAttentionSpec`，混合 skip 层时用 `lcm(tq_page, skip_page)`）；二是被排除层（skipped layer、sliding-window layer、Mamba layer）的 backend 选择曾可能被错误强制为 `TURBOQUANT`，现正确回退到默认 backend；三是 ROCm 上上游 `flash_attn_varlen_func` 不接受 `out=`，新增轻量 wrapper 检测该情形并仅在需要时拷贝结果。代码层面 `get_boundary_skip_layers` 由接收 `num_layers` 改为接收 `model_config`：hybrid 时关闭 boundary protection 并返回 `[]`（注释说明 hybrid 通常只有 8-12 个 full-attention 层，硬性 n=2 会覆盖约 40%，且 dense GSM8K 基线不适用），新增 `_get_full_attention_layer_indices` 覆盖 `layer_types`（Qwen3.5/Next）、`layers_block_type`（Jamba/Zamba2）、`attn_type_list`（Minimax）三种约定；`arg_utils.py` 相应移除 `is_hybrid` 时抛错逻辑。

## 关键改动

```text
vllm/model_executor/layers/quantization/turboquant/config.py: get_boundary_skip_layers 改收 model_config、hybrid 关闭 boundary protection、新增 _get_full_attention_layer_indices
vllm/engine/arg_utils.py: 移除 hybrid 模型的 NotImplementedError，改用新的 boundary 计算
vllm/platforms/interface.py: turboquant_ 分支改用 TQFullAttentionSpec / lcm 计算 page size
tests/quantization/test_turboquant.py: 新增 TestHybridAttentionIndices 回归测试
```

```text
+    if model_config.is_hybrid:
+        attn_indices = _get_full_attention_layer_indices(model_config)
+        if not attn_indices:
+            raise NotImplementedError(...)
+        logger.info("TQ hybrid: full-attention layers %s", attn_indices)
+        return []
...
+                attn_page_size_1_token = lcm(tq_page, skip_page)
```

## 性能声明

PR 正文未给出定量性能数据（pack 的 "perf claims detected in body" 为 "(none detected by regex)"）。正文 Summary 仅定性说明："Overall, this makes TurboQuant work reliably on hybrid architectures while preserving the current behavior and baselines for dense models." 未提供 benchmark 数字、模型/硬件/精度/workload 条件；PR 未注明测试条件。

## 验证与回归风险

diff 在 `tests/quantization/test_turboquant.py` 新增 `TestHybridAttentionIndices`（覆盖 `layer_types` 得 [2,4,5]、`layers_block_type` 得 [1,3]、`attn_type_list` 得 [1,3,4]、无 hybrid hints 返回空），并以 `_dense_model_config` 辅助保留 dense 模型的 boundary skip 测试（32 层得 ["0","1","30","31"] 等）。回归风险：正文称保持 dense 模型现有行为与基线；`config.py` 注释指出 hybrid 因 full-attention 层少而关闭 boundary protection（dense 上无该保护时 Qwen3-4B GSM8K 掉约 30 分）。正文未给出新增测试的运行结果，也未说明对其他模型、硬件或精度路径的实测回归数据。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
