# PR #46442: [Fix] unified-memory: Name the draft KV dtype flag in the fused-region mismatch error

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46442](https://github.com/sgl-project/sglang/pull/46442) |
| 作者 | ispobock |
| 合并日期 | 2026-10-25 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#45700 改写 fused draft KV dtype 不匹配报错时丢了 flag 名,weekly 套件测试匹配失败;报错文本重新点名该 flag 即修复。

## 问题现象
weekly 套件中的 `test_kv_cache_configurator_draft_gate.py` 不再能匹配到 fused draft KV dtype 不匹配的错误信息,测试失败。该不匹配错误在如下场景触发:draft 以 `auto` 方式解析出的 KV dtype 与其 fused region 当初 size 时所用 dtype 不一致。

## 根因
正文披露:PR #45700 改写了这条 fused draft KV dtype mismatch 的报错文案,把 flag 名从报错里删掉了,而上述测试按报错文本匹配,于是失配。即根因是报错文案回归,而非配置器逻辑本身出错。

## 修复方案
改 `python/sglang/srt/mem_cache/kv_cache_configurator.py` 的 `_fused_draft_from_target_buffer`,把 ValueError 文案从"must match the dtype used to size its fused region"改回点名 flag,并直接给出解法——显式设置 `--speculative-draft-kv-cache-dtype`,使 fused region 按draft 的 KV dtype 来 size:

```diff
-                    f"pages holds {region_kv_dtype}. The draft KV dtype must "
-                    "match the dtype used to size its fused region."
+                    f"pages holds {region_kv_dtype}. Set "
+                    "--speculative-draft-kv-cache-dtype explicitly so the region "
+                    "is sized in the draft's KV dtype."
```

只改报错字符串,判断逻辑与触发条件不变。

## 影响与回归风险
- 回归面:仅 `kv_cache_configurator.py` 中该条错误信息文本。运行时行为、内存规划逻辑均不变;唯一外部影响是下游若有其他按旧文案匹配的脚本/测试,同样需要以新文案为准(本 PR 只处理了 weekly 套件这一处)。

## 关联
- https://github.com/sgl-project/sglang/pull/45700(改写报错文案、丢失 flag 名的引入 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
