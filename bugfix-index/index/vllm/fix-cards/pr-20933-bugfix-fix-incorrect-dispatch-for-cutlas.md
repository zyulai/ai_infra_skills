# PR #20933: [Bugfix] Fix incorrect dispatch for CutlassBlockScaledGroupedGemm and DeepGEMM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20933](https://github.com/vllm-project/vllm/pull/20933) |
| 作者 | mgoin |
| 合并日期 | 2025-07-15 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

FP8 MoE 内核选择条件过宽,改为显式限定 DeepGEMM 仅 SM90/SM100、CUTLASS 仅 SM100。

## 问题现象

正文未描述具体复现现象或报错日志,仅说明这些内核的启用检查需要改为显式判断:DeepGEMM 当前只支持 SM90 或 SM100,CutlassBlockScaledGroupedGemm 只支持 SM100。也就是说,在不满足这些架构条件的 GPU 上,FP8 MoE 层可能错误地启用并分派到不支持的内核路径。

## 根因

正文披露的根因是:Fp8MoEMethod 中对 DeepGEMM 与 CutlassBlockScaledGroupedGemm 的支持检查不够显式,而两者实际的支持架构范围是——DeepGEMM 仅 SM90 或 SM100,CutlassBlockScaledGroupedGemm 仅 SM100。(由 diff 推断)原条件用的是 `has_device_capability(90)` / `has_device_capability(100)`,这是「不低于该算力」的判断,会把更高或不相关的架构(如非目标 SM 的更新卡)也判定为可用,导致分派到不支持这些内核的设备上。

## 修复方案

修改 `vllm/model_executor/layers/quantization/fp8.py` 中 `Fp8MoEMethod.__init__` 的内核使能逻辑:

- DeepGEMM 分支:条件由 `current_platform.has_device_capability(90)` 改为 `current_platform.is_device_capability(90)`(精确匹配 SM90);新增 SM100 分支,用 `is_blackwell_deep_gemm_used()` 判断并打 "Using DeepGemm SM100 kernels for Fp8MoEMethod." 日志后同样置 `self.allow_deep_gemm = True`。
- CutlassBlockScaledGroupedGemm 分支:条件由 `has_device_capability(100)` 改为 `is_device_capability(100)`(精确匹配 SM100);「模型非 block 量化」的提示由 `warning_once` 降级为 `debug_once`,减少无谓告警噪音。
- 顺带修正一处 warning 文案里多余的空格(" DeepGemm kernels" → "DeepGemm kernels")。

关键 diff 摘录:

```diff
-                  and current_platform.has_device_capability(90)):
+                  and current_platform.is_device_capability(90)):
                 logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
                 self.allow_deep_gemm = True
+            elif (current_platform.is_cuda()
+                  and is_blackwell_deep_gemm_used()):
+                logger.info_once("Using DeepGemm SM100 kernels for "
+                                 "Fp8MoEMethod.")
+                self.allow_deep_gemm = True
```

## 影响与回归风险

- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:仅影响 FP8 block 量化 MoE 模型在 CUDA 平台上的内核选择路径。SM90/SM100 设备行为不变;此前被误判启用的其他架构设备会回退到默认内核分支并打 "DeepGemm not supported on the current platform." 警告。SM100 上 DeepGEMM 的启用新增依赖 `is_blackwell_deep_gemm_used()` 的返回值,若该函数判断与实际支持不符,可能出现 SM100 上不再启用 DeepGEMM 的行为变化。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
