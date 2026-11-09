# PR #44694: [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44694](https://github.com/vllm-project/vllm/pull/44694) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-06-06 |
| labels | bug, qwen, ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
RMSNorm+quant 融合的 dtype 检查漏掉 fused_add_rms_norm 节点,bf16 输入配 fp32 权重被错误融合,Qwen3.5-FP8 启动即崩;扩展守卫。

## 问题现象
nightly「LM Eval Qwen3.5 Models (B200)」job(Buildkite ci build #70074)在 main 上开始失败:Qwen3.5-FP8(block-FP8)引擎启动即崩溃,block-FP8 kernel 以 `weight.scalar_type() == input.scalar_type()` 断言中止。回归由 #42646「[perf] Add gemma RMS AR fusion」引入(GemmaRMSNorm 被 Qwen3.5 / Qwen3-Next 使用)。

## 根因
正文披露:#42646 把 `GemmaRMSNorm.forward_native` 的 residual 路径改为发射 `ir.ops.fused_add_rms_norm(x, residual, weight, eps)`,其中 `weight = self.weight.float() + 1.0` 是 fp32,而 `x`/`residual` 是 bf16;融合 pass 的 `_rms_input_weight_dtype_match` 额外检查只看 `vllm_ir.rms_norm` 节点,对 `fused_add_rms_norm` 匹配直接返回 True,于是 dtype 不匹配仍被融合,向严格的 block-FP8 kernel 发出了 `weight=fp32 / input=bf16` 的调用。

## 修复方案
`vllm/compilation/passes/fusion/rms_quant_fusion.py` 扩展 `_rms_input_weight_dtype_match`,同时处理 `fused_add_rms_norm` 节点(input 取 `args[0]`,weight 取 `args[2]`),dtype 不匹配时跳过融合、回落 unfused 路径,与既有 `rms_norm` 行为一致:

```diff
 _RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
+_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
 ...
-            if isinstance(x, fx.Node) and isinstance(weight, fx.Node):
-                return x.meta["val"].dtype == weight.meta["val"].dtype
+        elif node.target == _FUSED_ADD_RMS_NORM_OP:
+            # fused_add_rms_norm(x, residual, weight, epsilon, variance_size)
+            x, weight = node.args[0], node.args[2]
+        else:
+            continue
+        if isinstance(x, fx.Node) and isinstance(weight, fx.Node):
+            return x.meta["val"].dtype == weight.meta["val"].dtype
```

## 影响与回归风险
- 影响窗口:#42646 合入起至本 PR 的 nightly/主流分支。
- 回归面:编译 fusion pass 的 RMSNorm+quant 融合判定——受影响的 bf16 输入/fp32 权重组合改为跳过融合走 unfused 路径(启动崩溃消除,可能有轻微性能回退);dtype 匹配的 fused_add_rms_norm 融合行为不变。验证:gsm8k correctness(Qwen3.5-35B-A3B-FP8-DEP2,configs/models-qwen35-blackwell.txt)通过。

## 关联
- [Buildkite ci build #70074](https://buildkite.com/vllm/ci/builds/70074#019e9470-dfa7-479f-b6aa-b78bcddcd4c7)(失败的 nightly job)
- #42646(引入回归的 perf PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
