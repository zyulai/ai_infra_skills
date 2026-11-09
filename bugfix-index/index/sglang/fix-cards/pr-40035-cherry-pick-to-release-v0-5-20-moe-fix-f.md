# PR #40035: [Cherry-pick to release/v0.5.20] [Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation (#39920)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40035](https://github.com/sgl-project/sglang/pull/40035) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2026-09-17 |
| labels | cherry-pick |
| 热度 | 👍 0 · 💬 0 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
flashinfer_trtllm 路径在未设 `gemm1_clamp_limit` 时静默丢弃 `swiglu_limit` 的 SwiGLU clamp,现回退用 `swiglu_limit` 作为 clamp 上限;cherry-pick 到 v0.5.20。

## 问题现象
本 PR 是 cherry-pick 工作流自动创建的(把 #39920 的 commit `aebae58b8c7894287a9ea8b5d844c65bbb58c9b2` 摘到 `release/v0.5.20`),正文除来源信息外无独立描述。从原始标题可知问题为:使用 flashinfer_trtllm MoE 路径时,由 `swiglu_limit` 约束(clamp)的 SwiGLU 激活被静默丢弃——即模型配置了激活 clamp 上限但该限制实际未生效,输出数值会偏出预期范围且不报错。正文未描述具体复现日志。

## 根因
本 PR 正文未披露根因(仅含 cherry-pick 元信息;原始 PR #39920 的描述未包含在 pack 中)。

(由 diff 推断)位于 `python/sglang/srt/layers/quantization/fp8.py` 的 `_prepare_flashinfer_trtllm_activation_params`(为每个 expert 一次性物化 TRT-LLM 的可选 SwiGLU 参数)原先直接取 `self.moe_runner_config.gemm1_clamp_limit` 作为 `gemm1_clamp_limit` 传给 TRT-LLM kernel;当用户只在 `moe_runner_config` 里配置了 `swiglu_limit` 而未单独设置 `gemm1_clamp_limit` 时,传出去的是 None,TRT-LLM kernel 拿不到 clamp 上限,SwiGLU 激活的 clamp 被静默丢弃。

## 修复方案
修改 `python/sglang/srt/layers/quantization/fp8.py` 中 `_prepare_flashinfer_trtllm_activation_params`:计算 `clamp_limit = gemm1_clamp_limit if gemm1_clamp_limit is not None else swiglu_limit`,再把 `clamp_limit` 传给 `("gemm1_clamp_limit", ...)` 参数项;`gemm1_alpha`、`gemm1_beta` 的处理保持不变。

关键 diff 摘录:

```diff
+        clamp_limit = (
+            self.moe_runner_config.gemm1_clamp_limit
+            if self.moe_runner_config.gemm1_clamp_limit is not None
+            else self.moe_runner_config.swiglu_limit
+        )
         for name, value in (
             ("gemm1_alpha", self.moe_runner_config.gemm1_alpha),
             ("gemm1_beta", self.moe_runner_config.gemm1_beta),
-            ("gemm1_clamp_limit", self.moe_runner_config.gemm1_clamp_limit),
+            ("gemm1_clamp_limit", clamp_limit),
```

即显式设置了 `gemm1_clamp_limit` 时优先用之,否则回退到 `swiglu_limit`。

## 影响与回归风险
- 影响窗口:正文未说明,不写(该修复 cherry-pick 进 `release/v0.5.20`)。
- 回归面:影响 flashinfer_trtllm MoE 路径的激活参数准备:凡配置了 `swiglu_limit` 但未设 `gemm1_clamp_limit` 的模型,TRT-LLM kernel 现在会真正执行 clamp,输出数值可能与修复前不同(修复前为未 clamp 的错误值);显式设置了 `gemm1_clamp_limit` 的行为不变。若某模型同时配置两个参数且期望语义不同,需确认优先级(现以 `gemm1_clamp_limit` 优先)。

## 关联
- #39920(来源 PR)
- 来源 commit `aebae58b8c7894287a9ea8b5d844c65bbb58c9b2`

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
