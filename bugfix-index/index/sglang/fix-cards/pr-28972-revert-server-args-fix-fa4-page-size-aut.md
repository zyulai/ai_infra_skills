# PR #28972: Revert "[server_args] fix FA4 page_size auto-force for combined --attention-backend fa4"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28972](https://github.com/sgl-project/sglang/pull/28972) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-23 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回滚 PR #28825:FA4 page_size=128 自动强制条件从「组合/decode/prefill 任一 fa4」收窄回仅 prefill fa4。

## 问题现象
正文仅一句话:Reverts sgl-project/sglang#28825,未描述被回滚改动引发的具体问题、报错或影响面。

## 根因
正文未披露根因(未说明为何回滚 #28825)。

## 修复方案
整体回滚 PR #28825 的两处改动:
1. `python/sglang/srt/server_args.py` 的 `_handle_attention_backend_compatibility`:FA4 的 page_size 自动强制条件由「`attention_backend`/`decode_attention_backend`/`prefill_attention_backend` 任一为 fa4」收窄回仅 `self.prefill_attention_backend == "fa4"`(非 MLA 模型且 SM100 时强制 page_size 128):
```diff
         if (
-            (
-                self.attention_backend == "fa4"
-                or self.decode_attention_backend == "fa4"
-                or self.prefill_attention_backend == "fa4"
-            )
+            self.prefill_attention_backend == "fa4"
             and not self.use_mla_backend()
             and is_sm100_supported()
         ):
```
2. `test/registered/unit/server_args/test_server_args.py`:删除 #28825 新增的 `TestFa4PageSizeAutoForce` 测试类(原覆盖「组合 `--attention-backend fa4` 也强制 page_size=128」与「显式 `--prefill-attention-backend fa4` 强制」两个用例)。

## 影响与回归风险
- 影响窗口:#28825 合并之后至本回滚前。
- 回归面:SM100 非 MLA 模型使用组合参数 `--attention-backend fa4`(或 `--decode-attention-backend fa4`)时,page_size 不再被自动强制为 128,用户需自行指定;若 FA4 kernel 实际要求 page_size 128,该场景可能报错或性能下降(这正是 #28825 试图自动处理的路径)。显式 `--prefill-attention-backend fa4` 路径行为不变。

## 关联
- https://github.com/sgl-project/sglang/pull/28825(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
