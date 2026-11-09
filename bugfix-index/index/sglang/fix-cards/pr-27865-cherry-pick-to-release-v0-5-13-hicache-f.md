# PR #27865: [Cherry-pick to release/v0.5.13] [hicache] Fix UnifiedRadixCache write-back load_back / eviction crashes (#27108)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27865](https://github.com/sgl-project/sglang/pull/27865) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2026-06-11 |
| labels | cherry-pick |
| 热度 | 👍 0 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
write_back 策略下 write_backup 失败仍继续驱逐到 host 导致崩溃;返回 0 时中止 demote(cherry-pick #27108)。

## 问题现象
正文未描述具体现象(cherry-pick 自动正文仅含元信息),仅标题指出:UnifiedRadixCache 在 write-back 的 load_back / eviction 路径上崩溃。本 PR 把源 PR #27108 的修复 commit `db061e97c006ae006d030dde3a411b07a204568c` cherry-pick 到 `release/v0.5.13`。

## 根因
正文未披露根因细节。(由 diff 推断)write_back 策略下 `_evict_device_leaf` 先调用 `write_backup(node, write_back=True)`,不检查返回值就继续 `writing_check` 与 `_evict_to_host`;若 write_backup 无法分配 host 池(返回 0),设备 leaf 仍被驱逐到 host,导致崩溃/状态不一致。新增测试的 docstring 也写明意图:「when write_backup cannot allocate host pool, _evict_device_leaf should not evict it to host」。

## 修复方案
`python/sglang/srt/mem_cache/unified_radix_cache.py` 的 `_evict_device_leaf`:接收 `write_backup` 返回值,`written == 0` 时直接 return(中止 demote,不再 evict to host);并在 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` 新增 `test_hicache_evict_device_leaf_aborts_demote_when_backup_fails`(mock write_backup 返回 0,断言节点未 evicted、device value 仍在、无 host_value,且直接调 `_evict_to_host` 会触发 AssertionError)。

```diff
-                self.write_backup(node, write_back=True)
+                written = self.write_backup(node, write_back=True)
+                if written == 0:
+                    return
                 self.writing_check(write_back=True)
                 self._evict_to_host(node, tracker)
                 return
```

## 影响与回归风险
- 回归面:仅影响 write_back 策略下设备 leaf 的驱逐路径:backup 失败时驱逐被跳过(节点保留在设备),从崩溃/不一致变为安全中止。注意:标题与源 PR 提及 write-back 的 load_back 与 eviction 两类崩溃,但本 pack 可见 diff 只含 eviction(write_backup)侧改动与测试,未见 load_back 侧改动;cherry-pick 是否完整以源 PR #27108 为准。

## 关联
- [#27108](https://github.com/sgl-project/sglang/pull/27108)(源 PR:Fix UnifiedRadixCache write-back load_back / eviction crashes)
- 源 commit `db061e97c006ae006d030dde3a411b07a204568c`

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
