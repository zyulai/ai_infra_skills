# PR #4185: Fix eagle hang issue for max_new_tokens=1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4185](https://github.com/sgl-project/sglang/pull/4185) |
| 作者 | ispobock |
| 合并日期 | 2025-03-07 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
EAGLE 在 max_new_tokens=1 时 batch_is_full 未复位,调度器不再取新请求导致服务挂起。

## 问题现象
用 EAGLE 投机解码跑 MMLU 基准会 hang。正文复现命令:
- 服务端:`python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf --speculative-algo EAGLE --speculative-draft lmsys/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --mem-fraction 0.7`
- 客户端:`python3 benchmark/mmlu/bench_sglang.py --nsub 100 --ntrain 5 --parallel 2000`

触发条件:MMLU 基准设置 max_new_tokens=1,此时批次只有 target extend 阶段。

## 根因
正文披露:当 max_new_tokens=1 时只有 target extend 阶段,extend 之后 `batch_is_full` 没有被更新(复位),从而阻止调度器从 waiting queue 取新的 prefill batch,服务表现为 hang。

## 修复方案
修改 `python/sglang/srt/managers/scheduler.py` 的 `get_next_batch_to_run`:在 `filter_batch()` 之前记录 batch size,若过滤后 batch 变小(有请求结束离场),把 `batch_is_full` 置回 False,使调度恢复从等待队列取新 batch。

```diff
+            last_bs = self.last_batch.batch_size()
             self.last_batch.filter_batch()
+            if self.last_batch.batch_size() < last_bs:
+                self.batch_is_full = False
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动位于调度器的通用函数 `get_next_batch_to_run` 内,凡走过 filter_batch 且 batch 缩小的路径(含 EAGLE 投机解码调度)都会在 batch 缩小时复位 batch_is_full;若原 batch_is_full 状态依赖旧的非复位行为,可能提前解除「批满」判定。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
