# PR #30428: [Chore] Fix torch precision warning

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30428](https://github.com/vllm-project/vllm/pull/30428) |
| 作者 | yewentao256 |
| 合并日期 | 2025-12-11 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

vLLM 用旧 TF32 API 触发 PyTorch 2.9+ 弃用告警,改用 `torch.backends.cuda.matmul.fp32_precision` 新 API。

## 问题现象

启动 vLLM 时(EngineCore worker 初始化,日志来自 `multiproc_executor.py`),每个 worker 打出大量 `UserWarning`:

> Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, ... will be deprecated after Pytorch 2.9.

触发点在 `torch/__init__.py:1617` 的 `_C._set_float32_matmul_precision(precision)`,即 vLLM 调用 `torch.set_float32_matmul_precision` 时由 torch 内部触发,日志中重复出现 8 次以上。

## 根因

PR 正文披露:vLLM worker(`vllm/v1/worker/gpu_worker.py`)通过旧 API `torch.set_float32_matmul_precision(precision)` 设置 FP32 matmul 精度,该旧式接口在 PyTorch 2.9 之后将被弃用并触发上述 UserWarning;修复方式是改用新 API `torch.backends.cuda.matmul.fp32_precision=precision`。

## 修复方案

- `vllm/v1/worker/gpu_worker.py`:把 `torch.set_float32_matmul_precision(precision)` 改为 `torch.backends.cuda.matmul.fp32_precision = precision`。
- `vllm/envs.py`:`VLLM_FLOAT32_MATMUL_PRECISION` 的可选值从 `["highest", "high", "medium"]`(默认 `"highest"`)改为 `["ieee", "tf32"]`(默认 `"ieee"`),注释说明 `"ieee"` 强制完整 IEEE FP32、`"tf32"` 启用 TensorFloat32 加速。
- `tests/v1/e2e/test_async_scheduling.py`:测试里 `VLLM_FLOAT32_MATMUL_PRECISION` 由 `"highest"` 改为 `"ieee"`。

```diff
-        torch.set_float32_matmul_precision(precision)
+        torch.backends.cuda.matmul.fp32_precision = precision
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围;告警信息本身提到旧 API 将在 PyTorch 2.9 之后弃用。
- 回归面:所有 GPU worker 的 FP32 matmul 精度设置路径;环境变量取值收敛为 `ieee`/`tf32`,原先设置 `highest`/`high`/`medium` 的用户配置将不再合法(需改为 `ieee` 或 `tf32`);默认值由 `highest` 变为 `ieee`,语义上同为全精度 FP32。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
