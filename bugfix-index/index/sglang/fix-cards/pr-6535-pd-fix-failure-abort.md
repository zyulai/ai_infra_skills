# PR #6535: [PD] Fix failure abort

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6535](https://github.com/sgl-project/sglang/pull/6535) |
| 作者 | ByronHsu |
| 合并日期 | 2025-05-23 |
| labels | (none) |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
PD 分离下 KV 传输失败的请求残留队列与 KV 泄漏;移除失败请求、修正 chunk_cache 释放量并加故障注入开关。

## 问题现象
正文仅说明「The previous PR had memory leak issue」(上一个 PR 存在内存泄漏问题),未描述具体现象、日志或复现步骤。结合正文列出的改动项,受影响场景是 PD(disaggregation)decode 侧请求 KV 传输失败(abort)时:失败请求残留在等待队列中不被移除,chunk_cache 在请求结束时未按正确数量释放 KV indices。正文未描述具体现象(无日志与复现命令)。

## 根因
正文只披露「上一个 PR 有内存泄漏问题」,未给出逐点根因分析。从 diff 看两处机制:1) `pop_transferred` 此前只移除传输成功的请求,因 abort 而 FINISH 的失败请求留在队列中无人清理;(由 diff 推断) 2) `chunk_cache.cache_finished_req` 用 `len(req.origin_input_ids) + len(req.output_ids) - 1` 计算 KV 释放量,decode server 上 `req.output_ids` 为空时该表达式少算(diff 注释:「For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids」),导致 KV indices 释放不足(由 diff 推断)。

## 修复方案
1. `python/sglang/srt/disaggregation/decode.py`:`DecodePreallocQueue` 构造函数新增 `scheduler`、`tree_cache` 依赖(scheduler.py `init_disaggregation` 处传入);`pop_transferred` 先扫描队列,对 `finished_reason` 为 `FINISH_ABORT` 的失败请求调用 `self.scheduler.stream_output` 输出结果并加入待移除集合:
```diff
+        # First, remove all failed requests from the queue
+        for i, decode_req in enumerate(self.queue):
+            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
+                self.scheduler.stream_output(
+                    [decode_req.req], decode_req.req.return_logprob
+                )
+                indices_to_remove.add(i)
```
2. `python/sglang/srt/mem_cache/chunk_cache.py` 的 `cache_finished_req`:释放量改为 `len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)`,保证 output_ids 为空时释放全部 origin_input_ids。
3. `python/sglang/srt/disaggregation/utils.py`:新增环境变量 `DISAGGREGATION_TEST_FAILURE_PROB`,大于 0 时按该概率把 `poll_and_all_reduce` 的 poll 结果强制置为 `KVPoll.Failed`,用于测试时模拟传输失败。
4. 重构:把 `ScheduleBatchDisaggregationDecodeMixin`(`prepare_for_prebuilt_extend`/`process_prebuilt_extend`)从 decode.py 迁出到新文件 `decode_schedule_batch_mixin.py`,schedule_batch.py 的导入同步更新。

## 影响与回归风险
- 影响窗口:正文未说明(提及「previous PR」,即该 PR 之前引入泄漏的变更窗口,未给具体版本)。
- 回归面:PD 分离 decode 请求的传输失败清理路径、chunk_cache 的请求结束释放逻辑(影响所有使用 ChunkCache 的场景)、`poll_and_all_reduce` 在设置了 `DISAGGREGATION_TEST_FAILURE_PROB` 时的行为(默认 0,不影响正常路径)、以及 `ScheduleBatchDisaggregationDecodeMixin` 的导入位置(纯重构,逻辑逐行迁出)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
