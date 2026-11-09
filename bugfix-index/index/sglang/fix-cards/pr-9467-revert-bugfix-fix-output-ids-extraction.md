# PR #9467: Revert "bugfix: Fix output_ids extraction in detokenizer_manager"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9467](https://github.com/sgl-project/sglang/pull/9467) |
| 作者 | JustinTong0323 |
| 合并日期 | 2025-08-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | regression |

## 一句话结论
回滚 #9047 的 output_ids 提取改动以修复其引入的 #9390;根因仍未解决,留待后续处理。

## 问题现象
正文未描述具体现象,仅说明 #9047 引入了需要回滚的问题(即 issue #9390),故整体回滚止损;作者同时写明「But still need future work to solve the root cause」,即根因尚未解决。

## 根因
正文未披露根因,并明确表示仍需后续工作解决根因,本次先回滚。(由 diff 推断)本回滚把 detokenizer_manager 下发的 output_ids 恢复为 `recv_obj.decode_ids`(#9047 曾改为 `recv_obj.output_ids`),并在 `entrypoints/context.py` 的两处 `append_output` 中恢复「找到首个 token 200006 并从该处截断」的临时 hack(#9047 曾移除),即整体恢复 #9047 合入前的行为。

## 修复方案
两处还原:
- `python/sglang/srt/managers/detokenizer_manager.py` 的 `handle_batch_token_id_out`:`output_ids` 取值由 `recv_obj.output_ids` 改回 `recv_obj.decode_ids`;
- `python/sglang/srt/entrypoints/context.py`:两处 `append_output` 恢复 token 200006 截断 hack,并还原文件头的 vLLM 出处注释。

```diff
-            output_ids=recv_obj.output_ids,
+            output_ids=recv_obj.decode_ids,

+            # TODO: REMOVE here:
+            # Very hacky, find the first occurrence of token 200006 and cut from there
+            try:
+                start_index = output_token_ids.index(200006)
+                output_token_ids = output_token_ids[start_index:]
+            except ValueError:
+                pass
```

## 影响与回归风险
- 回归面:影响 detokenizer_manager 向请求侧下发的 output_ids 来源(decode_ids 而非完整 output_ids),以及 context.py 两处 `append_output` 对 `output_token_ids` 的预处理(200006 截断会丢弃 200006 之前的 token);依赖 #9047 新行为的调用路径(需要完整 output_ids 的解析/上层消费方)回到旧行为。原作者注明根因未解决,后续仍可能重新修复,存在再次变更的风险。

## 关联
- [#9047](https://github.com/sgl-project/sglang/pull/9047)(被回滚的 PR)
- [#9390](https://github.com/sgl-project/sglang/issues/9390)(本回滚声称修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
