# PR #8333: Fix FP4 MoE accuracy from missing routed_scaling_factor

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8333](https://github.com/sgl-project/sglang/pull/8333) |
| 作者 | trevor-m |
| 合并日期 | 2025-07-25 |
| labels | (none) |
| 热度 | 👍 4 · 💬 4 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
routed expert 缩放因子在 FP4 MoE 路径未被应用(gsm8k 87% vs 应有 95%),在 flashinfer/cutlass 两条分支输出后补乘 `routed_scaling_factor` 修复。

## 问题现象
FP4 MoE(`nvidia/DeepSeek-R1-0528-FP4`,`quantization=modelopt_fp4`,tp_size=8,5-shot gsm8k,batch_size 512)精度异常:修复前 gsm8k exact_match 仅 0.8704(flexible-extract)/ 0.8681(strict-match);修复后 flashinfer cutlass moe 达 0.9492/0.9454,flashinfer cutlass + ep_moe 达 0.9545/0.9500,FP4 cutlass moe 达 0.9500/0.9477。对应 issue #7166。

## 根因
PR 正文披露:此前的 PR #6970 把 routed expert 缩放移入 `moe_sum_reduce` 并从模型 forward 中移除;但 `ModelOptNvFp4FusedMoEMethod` 代码路径不使用 `moe_sum_reduce`,导致 routed expert 缩放(`routed_scaling_factor`)在该路径上从未被施加,输出系统性偏小,表现为 FP4 MoE 精度下降。

## 修复方案
修改 `python/sglang/srt/layers/quantization/modelopt_quant.py` 的 `ModelOptNvFp4FusedMoEMethod.apply`,在两条分支(flashinfer moe 与 cutlass_moe_fp4)的输出后补乘缩放因子:

```diff
-            )
-            return output[0]
+            )[0]
+            if routed_scaling_factor is not None:
+                output *= routed_scaling_factor
+            return output
...
-        return cutlass_moe_fp4(
+        output = cutlass_moe_fp4(
...
-        ).to(x.dtype)
+        ).to(x.dtype)
+        if routed_scaling_factor is not None:
+            output *= routed_scaling_factor
+        return output
```

另在 `server_args.py` 中移除 flashinfer MoE 启用时强制 `disable_shared_experts_fusion = True` 的逻辑及其警告,使共享专家融合可与该后端共存。

## 影响与回归风险
- 影响窗口:自 PR #6970 将缩放移入 `moe_sum_reduce` 起至本修复,`ModelOptNvFp4FusedMoEMethod` 路径的输出均缺少 `routed_scaling_factor`。
- 回归面:`modelopt_quant.py` 中 flashinfer 与 cutlass 两条 FP4 MoE apply 分支的输出数值,凡 `routed_scaling_factor` 非 None 的模型(如 DeepSeek-R1 系列)输出会恢复正确量级;`routed_scaling_factor` 为 None 时行为不变。`server_args.py` 的改动解除共享专家融合禁用,启用 flashinfer MoE 的默认部署形态随之变化(共享专家重新融合),内存/性能特性可能改变。

## 关联
- Fixes sgl-project/sglang#7166
- 关联 PR:sgl-project/sglang#6970(引入缩放迁移、造成本 bug 的改动)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
