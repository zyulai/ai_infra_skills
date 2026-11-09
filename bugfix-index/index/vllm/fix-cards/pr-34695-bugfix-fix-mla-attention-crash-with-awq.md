# PR #34695: [Bugfix] Fix MLA attention crash with AWQ/GPTQ quantized models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#34695](https://github.com/vllm-project/vllm/pull/34695) |
| 作者 | haosdent |
| 合并日期 | 2026-03-13 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论

MLA 注意力代码访问 `kv_b_proj.weight`,而 AWQ/GPTQ 量化层只有打包的 `qweight`,启动即抛 AttributeError;加 hasattr 守卫并用 params_dtype 兜底。

## 问题现象

运行 MLA 架构模型并使用 AWQ/GPTQ 量化(如 `cyankiwi/GLM-4.7-Flash-AWQ-4bit`)时崩溃,报错 `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'`。该错误在 MLA 注意力代码访问 `.weight` 属性的路径上触发,影响所有使用 AWQ/GPTQ 权重量化的 MLA 模型;正文未给出具体版本号或完整堆栈。

## 根因

正文明确披露:MLA attention 代码在 3 处访问 `self.kv_b_proj.weight.dtype`,但 AWQ/GPTQ 量化的 `ColumnParallelLinear` 层把权重存为打包 int32 的 `qweight` 而非 `weight`,于是属性访问直接失败。原有代码只考虑了"未量化"与"FP8 量化"两种情形(两者都存在 `weight` 属性),没有覆盖 AWQ/GPTQ 这类 W4A16 打包权重方案。

## 修复方案

用 `hasattr(self.kv_b_proj, "weight")` 守卫全部 `.weight.dtype` 访问(正文称共两处改动):

1. `MLAAttention.__init__`(正文称 Line 406):ROCm fp4 BMM 判定的 `and` 链中加 `hasattr` 守卫,AWQ/GPTQ 时短路为 `False`——正文指出这是正确的,因为打包 int32 权重本就不能用于 fp4 BMM:

```diff
         self.is_aiter_triton_fp4_bmm_enabled = (
             rocm_aiter_ops.is_fp4bmm_enabled()
+            and hasattr(self.kv_b_proj, "weight")
             and self.kv_b_proj.weight.dtype == torch.bfloat16
         )
```

2. `MLACommonImpl._compute_prefill_context`(正文称 Lines 2357-2371):引入局部变量 `_kv_b_proj_w_dtype`,有 `weight` 时用 `weight.dtype`,否则回退到 `params_dtype`(`LinearBase` 上必有)——即模型的计算精度(如 bf16/fp16),正是 AWQ/GPTQ 层期望的输入 dtype,再据此决定 prefill 的 cast 行为:

```diff
+            _kv_b_proj_w_dtype = (
+                self.kv_b_proj.weight.dtype
+                if hasattr(self.kv_b_proj, "weight")
+                else self.kv_b_proj.params_dtype
+            )
+            if use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype():
+                kv_c_normed = kv_c_normed.to(_kv_b_proj_w_dtype)
```

作者用 mock 对象验证了 5 种量化场景(unquantized bf16/fp16、FP8 有/无 fp8 prefill、AWQ)的判定逻辑,全部通过。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:按 diff,改动集中在 `vllm/model_executor/layers/attention/mla_attention.py` 两处只读判定:未量化与 FP8 模型 `hasattr(weight)` 恒为 True,取值与 cast 行为和改动前完全一致;AWQ/GPTQ 模型 fp4 BMM 判定短路为 False(等价于"不支持"),prefill cast 到 `params_dtype`。若某个量化方案同时暴露 `qweight` 与 `weight` 两个属性,或其真实输入 dtype 与 `params_dtype` 不符,行为可能与预期不同。非 MLA 后端不受影响。

## 关联

- https://github.com/vllm-project/vllm/issues/34561 (Closes)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
