# PR #7874: [Bugfix] Fix #7592 vllm 0.5.4 enable_chunked_prefill throughput is slightly lower than 0.5.3~0.5.0.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7874](https://github.com/vllm-project/vllm/pull/7874) |
| 作者 | noooop |
| 合并日期 | 2024-09-02 |
| labels | ready |
| 热度 | 👍 3 · 💬 53 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
0.5.4 开启 chunked prefill 后 running 队列把新 prefill 排在 decode 前造成系统抖动、吞吐略降;修复为按 decode 优先的顺序重建 running 队列。

## 问题现象
vllm 0.5.4 开启 `enable_chunked_prefill` 时吞吐略低于 0.5.3~0.5.0(对应 issue #7592)。按官方文档的定义:默认 vLLM scheduler 优先 prefill,而一旦启用 chunked prefill,策略应改为优先 decode 请求。

## 根因
正文披露:优先 prefill 会引发并加剧系统抖动("Prioritizing prefill causes and aggravate system thrashing")。正文没有更深入的机制分析;从 diff 看,0.5.4 的 `_schedule_chunked_prefill` 在更新 `self.running` 队列时先追加新调度的 prefill、再追加 decode,使得 prefill 在 running 队列中排在 decode 之前,与「chunked prefill 下应优先 decode」的既定策略不符(由 diff 推断)。

## 修复方案
改 `vllm/core/scheduler.py` 的 `_schedule_chunked_prefill`:把 running 队列的追加顺序从「prefills → running decode → running prefill → swapped decode → swapped prefill」改为「swapped_in decode → swapped_in prefill → running decode → running prefill → prefills(最后)」,即最简单的「对 running 队列排序」,让已运行的 decode 与 swap-in 请求在后续调度中天然优先于新 prefill,同时完全不动 chunked prefill 本身的调度性能。另在 `tests/basic_correctness/test_chunked_prefill.py` 跳过一个 flaky 用例(fp8_e4m3 KV + chunked_prefill_token_size=4 的 Qwen2-1.5B-Instruct-FP8-K-V,引用 #7874、#8051)。

## 影响与回归风险
- 影响窗口:正文指出 0.5.4(相对 0.5.3~0.5.0)存在该回退,未给出其他版本范围。
- 回归面:仅 chunked prefill 调度路径中 running 队列的插入顺序;改变的是后续 step 的调度优先级,不影响单步调度输出构造。preempted 请求回 waiting 队列的逻辑不变。

## 关联
FIX #7592;测试跳过注释引用 #7874、#8051、#7378

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
