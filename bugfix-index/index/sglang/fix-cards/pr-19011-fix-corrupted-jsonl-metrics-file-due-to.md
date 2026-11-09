# PR #19011: Fix corrupted JSONL metrics file due to concurrent writes

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19011](https://github.com/sgl-project/sglang/pull/19011) |
| 作者 | talorabr |
| 合并日期 | 2026-02-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
并发写 metrics 导致 JSONL 文件损坏(行交错);`FileRequestMetricsExporter._write_record` 写文件全程加锁修复。

## 问题现象
使用 sglang server 参数 `--export-metrics-to-file --export-metrics-to-file-dir /tmp/sglang-metrics` 时,并发写会损坏 metrics 的 jsonl 文件。正文给出 32 并发请求下的损坏文件示例,损坏行形如:

```
, "id": "a9667994aff3400686d6d3eb95ee0235", "finish_reason": {"type": "stop", "matched": 128009}, "prompt_tokens": 3538, ...
```

即单行 JSON 被截断/交错,记录不再合法。正文说明「Write lock mechanism will prevent this from happening」,即写锁可避免该问题。

## 根因
正文披露的根因:多个并发写同时操作同一个 metrics jsonl 文件,导致文件内容损坏。未给出更底层的机制分析(如具体交错发生在 handler 写入还是 flush 阶段)。

## 修复方案
正文 Modifications:为 `FileRequestMetricsExporter` 增加 `RWLock`(写锁),在 `_write_record` 打开 metrics 文件时获取。diff 显示的实现为 `asyncio.Lock`:
- `__init__` 中新增 `self._current_file_lock = asyncio.Lock()`;
- `write_record` 中将「确保当前小时的 file handler 打开 → 判空返回 → `_format_output_data` 格式化 → `asyncio.to_thread(write_file)` 写入并 flush」的整个流程包进 `async with self._current_file_lock:`,使同事件循环内的写记录串行化。

```diff
+        self._current_file_lock = asyncio.Lock()
```

```diff
+            async with self._current_file_lock:
                 # Ensure correct file handler is open for current hour
                 self._ensure_file_handler(hour_suffix)
...
-            await asyncio.to_thread(write_file)
+                await asyncio.to_thread(write_file)
```

注:正文称新增的是 `RWLock`,diff 中实际为 `asyncio.Lock`。

## 影响与回归风险
- 回归面:`write_record` 的全流程(handler 轮换、格式化、写盘)被串行化,metrics 导出吞吐可能受影响;作者评估:不期望影响模型输出/精度,也不期望影响延迟/吞吐。跨进程/跨事件循环的并发写不在该锁保护范围内(由 diff 推断)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
