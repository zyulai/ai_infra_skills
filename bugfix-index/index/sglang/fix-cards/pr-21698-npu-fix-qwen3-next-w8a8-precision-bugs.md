# PR #21698: [npu]fix: qwen3-next w8a8 precision bugs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21698](https://github.com/sgl-project/sglang/pull/21698) |
| 作者 | ranjiewen |
| 合并日期 | 2026-04-27 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

[PR #19321](https://github.com/sgl-project/sglang/pull/19321) 改动致 NPU 上 qwen3-next w8a8 模型量化参数加载错误,补全各参数的 loader 并修复 Triton grid 越限。

## 问题现象

NPU 上 qwen3-next 运行 w8a8 模型失败,直接诱因是 `in_proj_qkvz` 的量化参数加载不正确。该回归由 [PR #19321](https://github.com/sgl-project/sglang/pull/19321) 引入。正文未给出具体报错日志。

## 根因

正文披露:PR #19321 的改动导致 `in_proj_qkvz` 的 w8a8 量化参数(weight、weight_scale、weight_offset)加载不正确。结合 diff(由 diff 推断):`_override_weight_loader` 此前只覆盖 `module.weight` 一个参数的 loader,其余量化附属参数(weight_scale_inv / weight_scale / input_scale / weight_offset)未被设置为新 loader,导致这些参数走默认加载逻辑而出错。另一独立问题:`fused_qkvzba_split_reshape_cat` 的 Triton 实现在 prefill 阶段 grid size 超过 65535 会报错,需要改用 NPU 算子。

## 修复方案

`python/sglang/srt/models/qwen3_next.py` 两处:

1. `_override_weight_loader` 从只处理 `module.weight` 扩展为遍历 `("weight", "weight_scale_inv", "weight_scale", "input_scale", "weight_offset")` 五个属性,存在的都把 `_weight_loader`(或 `weight_loader`)设为 `new_loader`;
2. NPU 上从 `sgl_kernel_npu.fla.utils` 导入 `fused_qkvzba_split_reshape_cat` 并覆写同名函数——正文说明 NPU 版 `fused_qkvzba_split_reshape_cat_npu` 支持 prefill 阶段,避免 Triton 算子 grid size 超 65535 报错。

关键 diff 摘录:

```diff
-        param = module.weight
-        if hasattr(param, "_weight_loader"):
-            param._weight_loader = new_loader
-        else:
-            param.weight_loader = new_loader
+        for attr_name in (
+            "weight",
+            "weight_scale_inv",
+            "weight_scale",
+            "input_scale",
+            "weight_offset",
+        ):
+            param = getattr(module, attr_name, None)
+            if param is None:
+                continue
+            if hasattr(param, "_weight_loader"):
+                param._weight_loader = new_loader
+            else:
+                param.weight_loader = new_loader
```

## 影响与回归风险

- 影响窗口:自 PR #19321 合入起至本修复;具体版本号正文未说明,不写。
- 回归面:qwen3-next 的权重加载路径(`_override_weight_loader` 现在会触碰同名模块上全部存在的量化参数,非 NPU 平台也受此遍历影响)与 NPU 上 `fused_qkvzba_split_reshape_cat` 的选择(仅 `_is_npu` 时覆写)。

## 关联

- [PR #19321](https://github.com/sgl-project/sglang/pull/19321)(引入回归的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
