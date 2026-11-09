# PR #25770: [Bug][Mamba] Fix LRU list reference cycle leak in mamba_radix_cache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25770](https://github.com/sgl-project/sglang/pull/25770) |
| 作者 | zjd0112 |
| 合并日期 | 2026-05-20 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 1 · 💬 2 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
LRU 双链表删除节点不清自身 prev/next,同批逐出的节点形成引用环等 gen-2 GC,RSS 线性增长;删节点时置空指针修复。

## 问题现象
hybrid Mamba 模型(如 Qwen3-Next)在持续负载下 scheduler RSS 线性增长(内存泄漏)。作者实测:Qwen3-Next、RTX 5090、TP=2,RSS 以约 3.7 GB/h 增长。诊断证据:`malloc_trim()` 仅释放 0.3%(排除 glibc 碎片);`flush_cache` 仅释放 0.1%(排除缓存占用);`gc.collect()` 立即释放 1.9 GB(确认引用环);`gc.set_debug(DEBUG_SAVEALL)` 显示环中有 17270 个 TreeNode + 17270 个 RadixKey + 17270 个 list + 17194 个 defaultdict。

## 根因
正文披露:`LRUList._remove_node` 更新邻居指针跳过被删节点,但不清除被删节点自身的 `prev`/`next`:
```python
def _remove_node(self, node):
    setattr(getattr(node, self.prv), self.nxt, getattr(node, self.nxt))
    setattr(getattr(node, self.nxt), self.prv, getattr(node, self.prv))
    # ⚠️ node.prev / node.next still dangling
```
当同一批逐出多个节点(evict_full / evict_mamba / _iteratively_delete_tombstone_leaf 的常见情形)时,被逐出节点之间的悬挂 LRU 指针构成两元素引用环,Python 引用计数无法回收,只能等 gen-2 GC,而持续负载下 gen-2 GC 过于稀少,内存持续累积。

## 修复方案
在 `_remove_node` 中删除节点后将其自身 prev/next 置空,打断引用环,使引用计数立即回收:
```diff
         setattr(
             getattr(node, self.nxt), self.prv, getattr(node, self.prv)
         )  # node.next.prev = node.prev
+        # Clear self pointers to break reference cycles among evicted nodes.
+        setattr(node, self.prv, None)
+        setattr(node, self.nxt, None)
```
同一修复应用到三处同构实现:`python/sglang/srt/mem_cache/mamba_radix_cache.py`、`swa_radix_cache.py`(两处均为 LRUList 属性版),以及 `unified_radix_cache.py`(`node.lru_prev[pt] = None` / `node.lru_next[pt] = None`)。

## 影响与回归风险
- 影响窗口:正文未说明;问题自 mamba/swa/unified radix cache 引入 LRU 双链表实现起即存在。
- 回归面:三种 radix cache(mamba、swa、unified)的 LRU 节点删除路径;置空指针只影响已被摘链的节点,不影响在链节点;后续若有代码在 `_remove_node` 之后仍读取被删节点的 prev/next,会从「悬挂旧值」变为 `None`,需留意此类用法(现有调用按链表语义应无此依赖)。

## 关联
- https://github.com/sgl-project/sglang/issues/21535 (正文标注 Related)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
