# PR #4170: Memory pool fix for upstream change about eagle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4170](https://github.com/sgl-project/sglang/pull/4170) |
| 作者 | xiezhq-hermann |
| 合并日期 | 2025-03-07 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
上游 #3986 重构 memory pool 后破坏 hierarchical cache,本 PR 修正 allocator 相关接口使其恢复可用。

## 问题现象
正文未描述具体现象,仅说明 #3986 对 memory pool 的改动「breaks hierarchical cache」,即分层缓存(HiRadixCache/HiCache)路径在该改动之后无法正常工作;其余已知问题由后续独立 PR 处理。

## 根因
正文披露:PR #3986 对 memory pool 做了大量改动(substantial change),破坏了 hierarchical cache,本 PR 修复其周边的接口问题(interfaces problem)。
(由 diff 推断)具体机制是:#3986 之后设备侧 KV cache 的分配/释放改经 `TokenToKVPoolAllocator` 管理,而 `HiCacheController`、`HiRadixCache` 仍直接持有并使用 `MHATokenToKVPool` 的 `alloc`/`free`,接口不匹配导致分层缓存的写穿/回读路径失效。

## 修复方案
以接口适配为主(正文称 "Primarily interface changes"):
- `python/sglang/srt/managers/cache_controller.py`:`HiCacheController.__init__` 改为接收 `token_to_kv_pool_allocator`,内部经 `get_kvcache()` 取设备池;`load`/`evict_device` 中的 `alloc`/`free` 改走 allocator:
  ```diff
  -        device_indices = self.mem_pool_device.alloc(len(host_indices))
  +        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
  ```
- `python/sglang/srt/mem_cache/hiradix_cache.py`:构造函数改收 allocator,`MHATokenToKVPoolHost` 改用 `allocator.get_kvcache()` 创建;`_evict_write_through_selective` 的 `free` 改走 allocator;`_match_prefix_helper` 由递归改为迭代实现(返回 `value, node`)。
- `python/sglang/srt/mem_cache/memory_pool.py`:`MHATokenToKVPoolHost` 默认 `host_to_device_ratio` 由 2.0 调为 3.0。
- `python/sglang/utils.py`:移除对 `ChatCompletionMessageContentPart` 的依赖与 `MsgContent` 类型别名。

## 影响与回归风险
- 回归面:所有启用 hierarchical cache(HiRadixCache/HiCache)的路径——设备/主机间 KV 写穿、回读(load back)与逐出逻辑;`TokenToKVPoolAllocator` 的 alloc/free 语义变化;`_match_prefix_helper` 重写影响前缀匹配行为;`utils.py` 类型移除影响引用 `MsgContent` 的调用方。正文未给出受影响版本范围。

## 关联
- https://github.com/sgl-project/sglang/pull/3986 (引入问题的改动)
- https://github.com/sgl-project/sglang/pull/4082 (Large TP sizes 修复)
- https://github.com/sgl-project/sglang/pull/4009 (MLA support 修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
