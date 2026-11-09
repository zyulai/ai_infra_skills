# PR #52704: [Bugfix][Quantization] Fix OCP MX MoE emulation silently skipping mxfp6 activation QDQ

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52704](https://github.com/vllm-project/vllm/pull/52704) |
| 作者 | xuebwang |
| 合并日期 | 2026-08-19 |
| labels | bug, quantization |
| 热度 | 👍 2 · 💬 3 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

OCP MX 仿真把 4 个 mxfp6 激活 scheme 映射为调度器不认识的 "mxfp6",静默跳过激活伪量化、精度虚高;改为映射到具体 dtype 并抽成可测函数。

## 问题现象

端到端对照:`w_mxfp4_a_mxfp6_e3m2` 与 weight-only `w_mxfp4` 用相同量化权重、激活、路由做两次 MoE 前向,前者本应伪量化激活、输出应不同,main 上两者却 bit-identical(max|diff| = 0.0),回归测试失败;这些 scheme 的精度数字因此偏乐观。修复后(E=8、topk=2、hidden=256、intermediate=256、64 tokens、bf16、MI325X)max|diff| = 1.875、mean shift 9.44%——即这些 scheme 本应建模的仿真误差。

## 根因

正文披露:`OCP_MXQuantizationEmulationTritonExperts` 在 `__init__` 里建 scheme→dtype 映射,受影响的 4 个 scheme(`w_mxfp4_a_mxfp6_e3m2`、`w_mxfp4_a_mxfp6_e2m3`、`w_mxfp6_e3m2_a_mxfp6_e3m2`、`w_mxfp6_e2m3_a_mxfp6_e2m3`)被映射成 `"mxfp6"`,而 `moe_kernel_quantize_input` 只按 `"mxfp6_e3m2"`/`"mxfp6_e2m3"` 分派;于是落入结尾的 `else: return A, A_scale`,以未量化激活运行,静默无报错。另外该链没有 `w_mxfp6_e2m3_a_fp8` 分支,`_quant_dtype` 未赋值,`quant_dtype` property 会抛 AttributeError。

## 修复方案

`vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py`:抽出模块级 `activation_quant_dtype(ocp_mx_scheme)`,把 4 个 mxfp6 scheme 分别映射到 `"mxfp6_e3m2"`/`"mxfp6_e2m3"`,fp8 系列(补上 `w_mxfp6_e2m3_a_fp8`)映射到 fp8 dtype,weight-only 返回 None,未知 scheme 抛 `NotImplementedError`;`__init__` 改为调用该函数。新增 dispatchable 测试与端到端 MoE 前向测试。

```diff
+    elif ocp_mx_scheme in {
+        OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2,
+        OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2,
+    }:
+        return "mxfp6_e3m2"
+    elif ocp_mx_scheme in {
+        OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3,
+        OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3,
+    }:
+        return "mxfp6_e2m3"
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:`OCP_MXQuantizationEmulationTritonExperts`(ROCm 仿真后端)的激活量化 dtype 选择:4 个 mxfp6 激活 scheme 从跳过 QDQ 变为真正伪量化(输出数值会变),`w_mxfp6_e2m3_a_fp8` 从 AttributeError 变为 fp8 路径,未知 scheme 从静默回落变为 NotImplementedError。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
