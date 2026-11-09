# PR #20453: [AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20453](https://github.com/sgl-project/sglang/pull/20453) |
| 作者 | Duyi-Wang |
| 合并日期 | 2026-03-13 |
| labels | documentation, deepseek, run-ci |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
MXFP4 主模型 + EAGLE MTP 在 MORI FP4/FP8 dispatch 下 CUDA graph capture 崩溃;`run_moe_core` 补 BF16 权重反量化,并新增 NEXTN 独立 dispatch 环境变量。

## 问题现象
使用 quark MXFP4 模型(如 `DeepSeek-R1-0528-MXFP4`)并启用 EAGLE 投机解码(MTP 层)与 MORI FP4/FP8 dispatch(`SGLANG_MORI_FP4_DISP=True` 或 `SGLANG_MORI_FP8_DISP=True`)时,MTP 层在 CUDA graph capture 期间崩溃:

```
RuntimeError: Unsupported kernel config for moe heuristic dispatch
```

此外 MTP 层与主模型共享同一 MORI dispatch 配置,无法独立控制其 dispatch 量化格式。

## 根因
正文披露:MTP 层的 MoE 权重是 BF16,而主模型的 MoE 权重是 MXFP4。MORI dispatcher 会把激活量化为 FP4/FP8 格式,但 `run_moe_core` 没有「FP4/FP8 激活 + BF16 权重」组合的处理,于是报 unsupported kernel config 错误。

## 修复方案
1. **`ep_moe/layer.py` 的 `run_moe_core` 支持 BF16 权重**:按 `self.w13_weight.dtype` 检测 `is_bf16_weights`;FP8 dispatch + BF16 权重时用 `upscale()` 把 FP8 激活反量化回 BF16,FP4 dispatch + BF16 权重时用 `upscale_mxfp4()` 反量化——BF16 权重不接受量化激活,必须先反量化;

```python
is_bf16_weights = self.w13_weight.dtype == torch.bfloat16
...
if is_quark_w4a4 or is_bf16_weights:
    # W4A4 or BF16 weights with FP8 dispatch: must dequant FP8->BF16 first.
    dispatch_a1 = upscale(...)
```

2. **`moriep.py` 新增独立 NEXTN dispatch 变量**:`SGLANG_MORI_NEXTN_FP8_DISP` / `SGLANG_MORI_NEXTN_FP4_DISP` 单独配置 MTP 层,未设置时回退主模型的 `SGLANG_MORI_FP8_DISP`/`SGLANG_MORI_FP4_DISP`;MTP 层经 `lru_cache` key 区分获得自己的 `mori_op` 实例(`maxsize` 2→4),`dispatch_a/b` 改用实例上缓存的 flags;
3. **`is_nextn` 沿 dispatch 链传播**:`deepseek_v2.py` 传给 MoE experts → `MoeRunnerConfig` 新增字段 → `FusedMoE` 传给 dispatcher → `MoriEPDispatcher` → `init_mori_op` → `_get_mori_dispatch_quant_flags`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:MORI EP dispatch 路径(`run_moe_core` 反量化分支扩展、`_get_mori_dispatch_quant_flags` 现按 `is_nextn` 选变量、`init_mori_op` 签名与缓存容量变化);未设置 NEXTN 新变量时回退原变量、行为保持;`docs/references/environment_variables.md` 同步新增两行文档。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
