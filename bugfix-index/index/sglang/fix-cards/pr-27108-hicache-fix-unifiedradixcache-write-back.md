# PR #27108: [hicache] Fix UnifiedRadixCache write-back load_back / eviction crashes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27108](https://github.com/sgl-project/sglang/pull/27108) |
| 作者 | libertyeagle |
| 合并日期 | 2026-06-11 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 9 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论

修复 UnifiedRadixCache write-back 两处崩溃:backup 失败仍驱逐、load_back 未锁 host 值被驱逐。

## 问题现象

write-back 策略下 `UnifiedRadixCache` 的两类崩溃:

1. `_evict_device_leaf` 触发 `assert node.backuped`(在 `_evict_to_host` 中):host 页分配失败(如多数 host 页被锁)时 `write_backup` 返回 0,但代码仍把节点当作已备份并执行 host 驱逐。
2. `assert cd.host_value is not None`(在 `full_component.py` 的 `build_hicache_transfers` 中):`load_back` 在 device 页不足时触发 device 驱逐,驱逐引发 `write_backup` 并可能驱逐 host 页,把待 load 回的节点的 host 值也驱逐掉了。典型场景:两个长请求,一个新在 GPU、一个在 host;host 侧请求再次命中触发 load back 时,GPU 侧请求被驱逐,`write_backup` 需要腾出 host 页。

## 根因

正文披露两处根因:

1. `_evict_device_leaf` 对 `write_backup` 的返回值不检查:`write_backup` 因 host 分配失败返回 0 时,节点并未备份,不能执行 `_evict_to_host`。`hiradix_cache.py` 的 `evict` 已有正确逻辑(`written > 0` 才计入 write_back_nodes 并断言 `node.backuped`),unified 实现未对齐。
2. `load_back` 没有锁定待加载节点的 host 值。`hiradix_cache.py` 靠 `inc_lock_ref` 无条件保护祖先节点(无论有无 device 值);而 unified cache 因支持 SWA,`full_component.py` 的 `acquire_component_lock` 会跳过没有 device 值的节点(且同样作用于 full_component),host 值因此失去保护。

## 修复方案

1. `python/sglang/srt/mem_cache/unified_radix_cache.py` 的 `_evict_device_leaf`:接收 `write_backup` 返回值,`written == 0` 时直接 return,不再执行 `writing_check` 与 `_evict_to_host`(与 hiradix_cache 逻辑对齐;作者给出的另一备选是直接丢弃 device 节点不备份)。

```diff
-                self.write_backup(node, write_back=True)
+                written = self.write_backup(node, write_back=True)
+                if written == 0:
+                    return
                 self.writing_check(write_back=True)
                 self._evict_to_host(node, tracker)
```

2. `load_back` 保护待加载节点的 host 值防止被驱逐,正文说明直接使用 `inc_host_lock_ref` / `dec_host_lock_ref` 实现。(pack 的 diff 仅包含第 1 处代码改动与对应单测;第 2 处改动的 diff hunks 未出现在 pack 中,此处按正文描述记录。)

测试:在 `test_unified_radix_cache_unittest.py` 新增两个单测(FULL / FULL+SWA / FULL+Mamba 配置下运行)——被 host 锁定的 leaf 在 host 驱逐扫描中存活且锁正确释放;`write_backup` 返回 0 时 `_evict_device_leaf` 将节点留在 device(含负向对照:直接调 `_evict_to_host` 会抛 AssertionError)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 write-back 策略下的 UnifiedRadixCache:backup 失败时节点保留在 device(可能减少可回收内存,但不再崩溃);load_back 期间 host 值被锁定,host 驱逐可用页变少。write_through 等其他策略与 hiradix_cache 路径不变。正文未提供 accuracy 测试结果。

## 关联

正文未列出 issue/PR 链接(列出原始提交 `032756d33`、`e7147dd8e`)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
