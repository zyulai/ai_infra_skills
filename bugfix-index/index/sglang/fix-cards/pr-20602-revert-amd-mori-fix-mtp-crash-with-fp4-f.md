# PR #20602: Revert "[AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars."

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20602](https://github.com/sgl-project/sglang/pull/20602) |
| 作者 | Fridge003 |
| 合并日期 | 2026-03-14 |
| labels | documentation, deepseek |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
整体回退 #20453:撤销 AMD MORI 的 NEXTN 专用 FP4/FP8 dispatch 量化选择与 is_nextn 贯穿改动。

## 问题现象
正文未描述具体现象——正文仅一句 "Reverts sgl-project/sglang#20453",无症状、复现或影响面说明。从标题可知被回退的 #20453 标题为 "[AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars",即本 PR 撤销的是那个针对 AMD MORI 后端 MTP(MTP/NextN 层)在 FP4/FP8 dispatch 下崩溃的修复及其新增的环境变量。

## 根因
正文未披露根因,未说明为何回退。由 diff 推断,本 PR 把 #20453 引入的以下改动全部撤销:1) 删除 `SGLANG_MORI_NEXTN_FP8_DISP`/`SGLANG_MORI_NEXTN_FP4_DISP` 两个 MTP 专用 dispatch 量化环境变量及其文档条目,`_get_mori_dispatch_quant_flags` 恢复为只读全局 `SGLANG_MORI_FP8_DISP`/`SGLANG_MORI_FP4_DISP`;2) 移除贯穿 `MoeRunnerConfig`、`fused_moe_triton/layer.py`、`moriep.py` dispatcher、`deepseek_v2.py` 的 `is_nextn` 参数;3) `run_moe_core` 中把 `is_quark_w4a4 or is_bf16_weights`、`is_fp8_quant or is_bf16_weights` 的 dequant 条件恢复为仅 `is_quark_w4a4`、`is_fp8_quant`;4) `dispatch_a/dispatch_b` 恢复为每次调用时读取量化 flag,`lru_cache` maxsize 由 4 回到 2。回退动机正文未说明。

## 修复方案
以回退方式恢复 #20453 合并前的行为,涉及文件:`docs/references/environment_variables.md`、`python/sglang/srt/layers/moe/ep_moe/layer.py`、`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`、`python/sglang/srt/layers/moe/moe_runner/base.py`、`python/sglang/srt/layers/moe/token_dispatcher/moriep.py`、`python/sglang/srt/models/deepseek_v2.py`。

关键摘录(moriep.py 恢复全局 env 读取):
```diff
-@lru_cache(maxsize=4)
-def _get_mori_dispatch_quant_flags(is_nextn=False):
-    fp8_var, fp4_var = "SGLANG_MORI_FP8_DISP", "SGLANG_MORI_FP4_DISP"
-    if is_nextn and (
-        "SGLANG_MORI_NEXTN_FP8_DISP" in os.environ
-        or "SGLANG_MORI_NEXTN_FP4_DISP" in os.environ
-    ):
-        fp8_var, fp4_var = "SGLANG_MORI_NEXTN_FP8_DISP", "SGLANG_MORI_NEXTN_FP4_DISP"
+@lru_cache(maxsize=2)
+def _get_mori_dispatch_quant_flags():
+    fp8_dispatch = get_bool_env_var("SGLANG_MORI_FP8_DISP", "False")
+    fp4_dispatch = get_bool_env_var("SGLANG_MORI_FP4_DISP", "False")
```

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围(#20453 合并后至本回退前)。
- 回归面:AMD MORI EP dispatch 的量化路径恢复旧逻辑——MTP/NextN 层不能再通过 NEXTN 专用环境变量单独选择 FP8/FP4 dispatch,与主网层共用全局变量;BF16 权重在 FP8/FP4 dispatch 下不再自动 dequant(#20453 添加的分支被移除),该组合下的行为回到 #20453 之前状态;`is_nextn` 参数从多个 MoE 接口签名中删除,调用方需与旧签名匹配。原 #20453 针对的 MTP crash 场景在回退后可能重新暴露(正文未说明替代方案)。

## 关联
- Reverts https://github.com/sgl-project/sglang/pull/20453

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
