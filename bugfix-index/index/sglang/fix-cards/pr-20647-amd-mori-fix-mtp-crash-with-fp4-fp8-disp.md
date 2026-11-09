# PR #20647: [AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20647](https://github.com/sgl-project/sglang/pull/20647) |
| 作者 | Duyi-Wang |
| 合并日期 | 2026-03-17 |
| labels | documentation, deepseek, run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
MTP 层 BF16 权重遇上 MORI FP4/FP8 dispatch 量化激活时无对应 kernel,CUDA graph 采集期崩溃;在 run_moe_core 中反量化回 BF16 并为 NEXTN 增加独立 dispatch 环境变量。

## 问题现象
使用 quark MXFP4 模型(如 `DeepSeek-R1-0528-MXFP4`)并开启 EAGLE 投机解码(MTP 层),同时启用 MORI FP4/FP8 dispatch(`SGLANG_MORI_FP4_DISP=True` 或 `SGLANG_MORI_FP8_DISP=True`)时,MTP 层在 CUDA graph 采集期间崩溃:

```
RuntimeError: Unsupported kernel config for moe heuristic dispatch
```

触发条件为「MXFP4 主模型 + MTP 层 + MORI FP4/FP8 dispatch」三者同时存在;另外 MTP 层此前只能与主模型共用同一套 MORI dispatch 配置,无法独立控制其 dispatch 量化格式。

## 根因
正文披露:MTP 层的 MoE 权重是 BF16,而主模型的 MoE 权重是 MXFP4。MORI dispatcher 会把激活量化成 FP4/FP8 格式,但 `run_moe_core` 没有处理「FP4/FP8 激活 + BF16 权重」的组合,导致抛出不支持的 kernel 配置错误。次要问题:MTP 层与主模型共享同一份 MORI dispatch 配置,无法独立设置 dispatch 量化格式。

## 修复方案
1. `ep_moe/layer.py` 的 `run_moe_core` 新增 `is_bf16_weights` 检测(基于 `self.w13_weight.dtype`):FP8 dispatch + BF16 权重时用 `upscale()` 把 FP8 激活反量化回 BF16;FP4 dispatch + BF16 权重时用 `upscale_mxfp4()` 把 FP4 激活反量化回 BF16:

```diff
+        is_bf16_weights = self.w13_weight.dtype == torch.bfloat16
...
-            if is_quark_w4a4:
+            if is_quark_w4a4 or is_bf16_weights:
...
-            if is_fp8_quant:
+            if is_fp8_quant or is_bf16_weights:
```

2. `moriep.py` 新增 `SGLANG_MORI_NEXTN_FP8_DISP` / `SGLANG_MORI_NEXTN_FP4_DISP` 环境变量,未设置时回退到主模型的 `SGLANG_MORI_FP8_DISP`/`SGLANG_MORI_FP4_DISP`;MTP 层经 `lru_cache` key 区分获得自己的 `mori_op` 实例。
3. `is_nextn` 全链路传递:`deepseek_v2.py` 传入 MoE experts → `moe_runner/base.py` 的 `MoeRunnerConfig` 增加 `is_nextn` 字段 → `fused_moe_triton/layer.py` 的 `FusedMoE` 传给 dispatcher → `moriep.py` 的 `MoriEPDispatcher` → `_MoriEPDispatcherImplBase` → `init_mori_op` → `_get_mori_dispatch_quant_flags`。

另注:本 PR 是旧 PR #20453 的新版本;为修复 DeepEP CI 失败,仅在 `a2a_backend.is_mori` 时才把 `is_nextn` 传给 `MaybeTboDeepEPDispatcher`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动涉及 `ep_moe/layer.py`(`run_moe_core` 反量化分支)、`moriep.py`(dispatch flag 读取与 `init_mori_op` 签名)、`fused_moe_triton/layer.py`、`moe_runner/base.py`、`deepseek_v2.py` 与环境变量文档。影响使用 MORI a2a backend 的 MoE dispatch 路径:BF16 权重 + FP4/FP8 dispatch 现在会先反量化,精度与性能可能与原量化路径不同;`_get_mori_dispatch_quant_flags`/`init_mori_op` 的 lru_cache 从 2 扩到 4 并改签名,影响所有 MORI dispatcher 调用方;`is_nextn` 仅在 mori backend 下传入 DeepEP dispatcher,其他 a2a backend 行为不变。

## 关联
- sgl-project/sglang#20453(旧版本 PR,正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
