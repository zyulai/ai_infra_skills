# PR #8533: fix incorrect increase of hit count

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8533](https://github.com/sgl-project/sglang/pull/8533) |
| 作者 | huangtingwei9988 |
| 合并日期 | 2025-07-31 |
| labels | ready-to-merge |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
HiRadixCache 的 hit count 在多种场景被重复累加而虚高,导致大量无价值 cache 被 offload、成本上升;移除前缀匹配阶段的重复累加。

## 问题现象
HiRadixCache(HIRO radix cache)的 hit count 会在以下场景不正确地累加,虚高的 hit count 使 cache 层把大量无价值(worthless)的 cache offload 出去,导致成本增加:
- 长输入请求被切成多个 chunk(chunked_req)时,每个 chunk prefill 完成后都会调用 `cache_unfinished_req`,hit count 被多累加 chunks 次;
- 开启 `enable_hicache_storage` 时,请求会调用两次 `init_next_round_input`,hit count 被错误累加;
- 同一请求的 hit count 会在 `_match_prefix_helper` 和 `_insert_helper` 两处都被增加,而(作者质疑)增加一次也许就够。
正文未给出具体数字或日志。

## 根因
正文披露:hit count 的累加逻辑存在重复计数——chunked prefill 每个分块都触发一次累加、hicache storage 使 `init_next_round_input` 双调用、前缀匹配与插入两个阶段各累加一次。具体哪一处是"权威"计数点,正文未裁决,仅在第三点提出"maybe just once is enough?"的疑问。

## 修复方案
`python/sglang/srt/mem_cache/hiradix_cache.py` 的 `_match_prefix_helper`:删除两处 `inc_hit_count` 调用(节点被 split 出的 `new_node` 分支与完整匹配的 `child` 分支),前缀匹配阶段不再累加 hit count:

```diff
             if prefix_len < len(child.key):
                 new_node = self._split_node(child.key, child, prefix_len)
-                self.inc_hit_count(new_node)
                 if not new_node.evicted:
                     value.append(new_node.value)
                 node = new_node
                 break
             else:
-                self.inc_hit_count(child)
                 if not child.evicted:
                     value.append(child.value)
                 node = child
```

注意:pack 提供的 diff 仅含上述 `_match_prefix_helper` 改动,正文所列 chunked_req 与 `enable_hicache_storage` 两个场景的对应修复在提供的 diff 中不可见。

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:仅 `hiradix_cache.py` 的前缀匹配路径;匹配阶段不再抬高节点 hit count,HiRadixCache 的 offload/eviction 决策所依据的 hit 统计随之变化(这正是修复目的);hit count 的累加只剩插入侧(按 diff 陈述)。若下游策略依赖匹配阶段也累加,行为会有差异(正文未讨论)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
