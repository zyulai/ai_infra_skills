# PR #38719: [Fix] moe: Preserve FP32 router precision across model gates

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38719](https://github.com/sgl-project/sglang/pull/38719) |
| 作者 | seongyun1104 |
| 合并日期 | 2026-11-02 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 20 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论

五个 MoE 模型在模型专属边界丢弃 router 精度:correction bias 被 bf16 构建、fp32 gate 输出在 top-k 前被降精度;统一改为保持 fp32,消除 checkpoint bias 塌缩与精度损失。

## 问题现象

正文未描述具体运行时故障现象或日志;问题定位为若干 MoE 模型在 model-specific 边界上丢弃 router 精度的代码路径(继承 #38695 中指出的待独立修复项):专家选择依赖 fp32 敏感的 score 运算,而这些路径要么让 checkpoint 的 correction bias 数值塌缩,要么在 expert selection 前一刻丢掉 fp32 router 输出。

## 根因

正文披露两类根因,均属模型文件内的 dtype 处理错误:

1. MiMo-V2、Ernie4、Kimi Linear 的 `e_score_correction_bias` 可能在 bf16 中构建(如 MiMo-V2 在 `modelopt_fp4` 量化 + flashinfer_trtllm runner 时显式选 bf16),而 expert selection 以 fp32 敏感的 score 运算消费它,导致 checkpoint 中的 bias 值被塌缩(collapse)。
2. Bailing MoE 与 Bailing MoE Linear 在配置为 fp32 router weights 时用 fp32 计算 gate GEMM,却在 top-k 之前把 logits `.to(hidden_states.dtype)` 转回 hidden-state dtype,使 fp32 router 输出被立刻丢弃。

正文另说明:LLaDA2 原本也有同样的 router 输出降精度,但已由 #31768 独立修复,故本 PR 不动它。

## 修复方案

- MiMo-V2、Ernie4、Kimi Linear:`e_score_correction_bias` 参数一律以 `dtype=torch.float32` 构建;MiMo-V2 中按量化配置降级到 bf16 的分支被整体删除。
- Bailing MoE / Bailing MoE Linear:gate forward 直接返回 router-weight dtype 的 GEMM 输出,删除 top-k 前的 `.to(hidden_states.dtype)`。

关键摘录(bailing_moe.py):

```diff
-        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
-            hidden_states.dtype
-        )
-        return logits
+        return F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
```

patch 只改 5 个模型文件(bailing_moe.py、bailing_moe_linear.py、ernie4.py、kimi_linear.py、mimo_v2.py);不引入共享 `RouterGate`,不加 kernel,不修改 expert 计算。未新增单测,遵循 #38695 中维护者关于此类模型文件修复不需要单测的指导。

## 影响与回归风险

- 影响窗口:正文未说明具体版本范围,不写。
- 回归面:按 diff,仅上述五个模型的 router 数值路径——correction bias 参数 dtype 由(默认随 hidden dtype / 条件 bf16)统一为 fp32,gate logits 不再转回 hidden dtype。影响这五个模型的 top-k/expert 选择数值结果;Bailing 在 fp32 router 配置下少一次 dtype 转换(性能微正)。不涉及其他模型、kernel 或 expert 计算。验证仅到字节码编译、pre-commit 与 git diff --check,无精度基准数据。

## 关联

- PR [#38695](https://github.com/sgl-project/sglang/pull/38695)(本 PR 响应其中指出的独立模型修复)
- PR [#31768](https://github.com/sgl-project/sglang/pull/31768)(已修复 LLaDA2 同类问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
