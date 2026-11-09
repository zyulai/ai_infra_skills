# PR #22832: [sgl] fix incorrect behavior in cuda graph draft extend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22832](https://github.com/sgl-project/sglang/pull/22832) |
| 作者 | 2022tgoel |
| 合并日期 | 2026-04-20 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 6 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
decode CUDA graph 未运行时(如 batch 过大)draft extend 仍取 graph 缓冲致崩溃/越界;按是否走 graph 选正确张量修复。

## 问题现象
正文未描述具体错误日志;指出 decode CUDA graph 可能没有运行(例如 batch size 过大时),此时 `multi_layer_eagle_worker_v2` 的 `_draft_extend_for_decode` 仍无条件从 graph runner 的缓冲区取 hidden states 等张量,会引发崩溃 / 越界内存访问(正文原话:"while this prevents crashing / OOB memory access")。

## 根因
正文披露:decode CUDA graph 可能没有运行(比如 batch size 太大),而原代码不考虑这一点,总是取 `cuda_graph_runner_for_draft_extend.get_last_runner()` 的缓冲区张量来更新 `req_to_hidden_states_pool`;graph 未运行时这些缓冲不是本轮的真实数据。修复思路即「根据 CUDA graph 是否运行来抓取正确的张量」。作者同时表示对现有逻辑存疑:`chain_mtp_hidden_states` 里每个 MTP step 的 hidden states 都不同,只存一份并不完全正确——本修复只是防止崩溃 / 越界,并非完全正确的方案。

## 修复方案
`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` 的 `_draft_extend_for_decode` 中,更新 `req_to_hidden_states_pool` 前按 `can_cuda_graph` 分支取数:
```python
if can_cuda_graph:
    last_runner = self.cuda_graph_runner_for_draft_extend.get_last_runner()
    hidden_states = last_runner.buffers.hidden_states
    ...
else:
    hidden_states = draft_logits_output.logits_output.hidden_states
    req_pool_indices = forward_batch.req_pool_indices
    extend_seq_lens = forward_batch.extend_seq_lens
    extend_start_loc = forward_batch.extend_start_loc
```
即:CUDA graph 跑了就用 graph runner 缓冲,没跑就从 `draft_logits_output` 与 `forward_batch` 取真实张量,再交给 `assign_hidden_states_pool_triton`。正文未提供准确性/速度测试结果(Checklist 各项勾选但对应小节为空)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:EAGLE 多层 worker v2 的 decode draft extend 路径。graph 命中时行为不变;graph 未命中(batch 过大等)时从 forward batch 取值,影响 KV cache 回退所用的 hidden states 来源。作者在正文明确提示该逻辑可能仍不完全正确(每步 hidden states 覆盖问题),属于「防崩溃」性质的修复。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
