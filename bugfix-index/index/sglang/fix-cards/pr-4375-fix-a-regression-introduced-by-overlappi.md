# PR #4375: Fix a regression introduced by overlapping KV cache writing

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4375](https://github.com/sgl-project/sglang/pull/4375) |
| 作者 | merrymercy |
| 合并日期 | 2025-03-13 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
#4356 引入的 KV cache 重叠写入在大 batch 下拖慢吞吐,现改为仅小 batch(shape[0]<4)走重叠路径。

## 问题现象
正文仅说明 #4356 引入了 throughput 上的性能回归(Fix a perf regression on throughput),未描述具体日志或复现数据。

## 根因
正文披露:回归由 #4356(overlapping KV cache writing)引入。
(由 diff 推断)机制:`set_kv_buffer` 中 `capture_mode` 分支原本无条件把 K/V 写入放到 alt_stream 上与当前流重叠,重叠带来的 stream 同步开销在写入规模较大时超过收益,导致吞吐下降;修复后仅当 `cache_k.shape[0] < 4`(极小 batch)时才走重叠路径,其余回到当前流直接写入。

## 修复方案
改 `python/sglang/srt/mem_cache/memory_pool.py` 的 `set_kv_buffer`:给重叠写入分支加上尺寸门槛,关键 diff(共 1 行改动):
```diff
-        if self.capture_mode:
+        if self.capture_mode and cache_k.shape[0] < 4:
             self.alt_stream.wait_stream(torch.cuda.current_stream())
             with torch.cuda.stream(self.alt_stream):
                 self.k_buffer[layer_id][loc] = cache_k
```
即 batch 维度 ≥4 时不再使用 alt_stream 重叠写,直接在当前 stream 写入 KV buffer。

## 影响与回归风险
- 影响窗口:正文未说明;按引入关系,回归窗口为 #4356 合并之后至本 PR 之前。
- 回归面:所有 `capture_mode`(CUDA graph capture)下经 `set_kv_buffer` 的 KV 写入路径;shape[0] < 4 的写入仍走 alt_stream 重叠,行为不变;shape[0] ≥ 4 的写入恢复为同步写,若上层依赖重叠语义的时序需留意(写仍在当前 stream 上完成,后续同 stream 读取天然有序)。

## 关联
正文仅以编号提及引入回归的 #4356(未附链接;对应 https://github.com/sgl-project/sglang/pull/4356)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
