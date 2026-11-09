# PR #18013: [BugFix] fix gpt-oss accuracy issue when enabling piecewise cuda graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18013](https://github.com/sgl-project/sglang/pull/18013) |
| 作者 | zminglei |
| 合并日期 | 2026-02-01 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

启用 piecewise cuda graph 时 gpt-oss 精度异常(GSM8K 0.785→0.845 修复):decode 阶段跳过 SWA token 驱逐。

## 问题现象

启用 piecewise cuda graph 后 gpt-oss 出现精度问题。作者以 GSM8K 200 题基准对比:修前 Accuracy 0.785(Invalid 0.010),修后 Accuracy 0.845(Invalid 0.015),吞吐量基本持平(2590.9 → 2478.2 token/s)。其余触发细节正文未展开,issue #17839 有报告。

## 根因

正文除标题外未披露根因分析(Modifications 一节为空)。(由 diff 推断)改动位于 `schedule_batch.py` 的 `maybe_evict_swa`:在 `forward_mode.is_decode()`、`server_args.enable_piecewise_cuda_graph` 开启且 `tree_cache` 不是 chunk cache 时,直接提前 `return`,跳过本轮 SWA(sliding window attention)token 驱逐——暗示此前 decode 阶段在 piecewise cuda graph 启用时驱逐 SWA token 会破坏精度(与 piecewise cuda graph 对 batch 的处理交互);同时把 `server_args = get_global_server_args()` 的获取提前到函数开头以支撑新增判断。

## 修复方案

改 `python/sglang/srt/managers/schedule_batch.py` 的 `maybe_evict_swa`,新增提前返回:

```diff
+            server_args = get_global_server_args()
+
+            if (
+                self.forward_mode.is_decode()
+                and server_args.enable_piecewise_cuda_graph
+                and not self.tree_cache.is_chunk_cache()
+            ):
+                return
+
             for idx, req in enumerate(self.reqs):
```

即 decode 模式 + piecewise cuda graph 开启 + 非 chunk cache 时,不再执行 SWA 驱逐逻辑;chunked prefill 路径中原有的 `server_args` 获取随之移到函数开头。

## 影响与回归风险

- 回归面:所有启用 SWA(radix cache 支持 sliding window)模型的 decode 调度路径,在 `--enable-piecewise-cuda-graph` 下将不再于 decode 阶段驱逐 SWA token,可能改变 KV/内存回收时机(未测内存影响,正文未提供);chunked prefill 与非 piecewise 路径行为不变。

## 关联

- Fixes https://github.com/sgl-project/sglang/issues/17839

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
