# PR #66855: [CPU][Perf] Split oversized final attention spans

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tianmu-li |
| 合并 | 2026-11-13 |
| 分类 | Attention / Kernel(tier: major) |
| Label | cpu |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66855 |

## 一句话结论

CPU attention 的 final partition 改为逐 workitem 发射,使 oversized decode remainder 可被多个 worker 分摊:batch 128、95 threads 下 attention latency 4.31 → 0.92 ms(4.67x),平衡 batch 与固定输入输出保持不变。

## 动机

final partition 接收此前各 partition budget 剩下的全部 work,把多个 request 串行进同一个 task,无法跨 worker 分摊;batch 128 时出现四个 34-workitem 的 span,单 task 序列化拖慢整体。open PR 检索(CPU attention、scheduling、task spans)未发现重复修复。

## 优化点

- 复用既有 singleton 发射路径,把 final partition 的无界 remainder 作为可共享 task 发射:发射条件追加 `thread == thread_num - 1`(见摘录),使最后一个 thread 上的 remainder 逐 workitem 拆分。
- 单个大 prefill workitem 保持不可拆分。
- Tradeoff(正文明示):额外的 task dispatch 对小 workload 有开销,尤其是单线程时——那时每个 workitem 都属于 final partition。
- 分区、KV splits、reduction 与 scratchpad 尺寸保持不变。

## 关键改动

```text
csrc/cpu/cpu_attn_impl.hpp: AttentionScheduler final-partition 发射条件追加 thread == thread_num - 1
tests/kernels/attention/test_cpu_attn.py: 新增 decode_final_partition_remainder 参数化 case((1,4096) x 6,heads 32/4)
```

```cpp
// Emit the final partition's unbounded remainder as shareable tasks.
if ((item.group > 1 && request_q_token_num(item.req_id) > 1) ||
    thread == thread_num - 1) {
  if (start >= 0) {
    emit(start, idx);
    start = -1;
```

## 性能声明

- batch 128、95 threads:attention latency 4.31 → 0.92 ms(4.67x throughput)。
- Decode benchmark 环境:BF16 AMX,context 4096,heads 32/4,head size 128,block size 32,KV splitting enabled,95 threads;结果为两次 per-run median 的平均,交替 baseline/candidate 顺序,每次 run 五次 warmup、100 次测量,使用 `benchmarks/kernels/cpu/benchmark_cpu_attn.py`。逐 batch:64 为 0.614 → 0.579 ms(1.06x);128 为 4.310 → 0.923 ms(4.67x);190 为 1.321 → 1.300 ms(1.02x,平衡 batch 对照,无测量到的回归);192 为 1.594 → 1.310 ms(1.22x)。计时只覆盖 attention 执行,不含 metadata 生成与 cache packing。
- 提交者明确:不声称端到端加速(no end-to-end speedup is claimed)。

## 验证与回归风险

- `pytest tests/kernels/attention/test_cpu_attn.py -q -k test_amx_spec_decode_gqa_correctness`:baseline 与 candidate 各 12 passed,覆盖 BF16/FP8 与两种 KV split 模式。
- `pytest ... -k "test_varlen_with_paged_kv_normal_amx and None-128-128-num_heads1 and (seq_lens1 or seq_lens2)"`:六个 mixed/prefill case 通过。
- batch 128、95 threads 下 metadata 从 380 spans(含四个 34-workitem span)变为 512 个 single-item span;检查确认 exactly-once 覆盖,workitem 与 reduction 记录不变。
- 固定输入的 decode 与 prefill attention 输出逐 bit 一致。
- CPU `LLM` 验证:Qwen2.5-1.5B-Instruct(BF16,95 threads,1024 input tokens,八个 greedy output tokens,batch 64/128)全部 192 请求通过 `check_logprobs_close`,188 条 token 序列完全一致;baseline 重跑同样出现 greedy 差异(六条序列)并通过同一检查,说明差异属基线自身波动。
- `pre-commit run --files csrc/cpu/cpu_attn_impl.hpp tests/kernels/attention/test_cpu_attn.py` 与 commit hooks、`git diff --check` 均通过;改动经提交者 review。
- 风险:小 workload 与单线程场景下额外 task dispatch 可能带来开销(正文已声明);单个大 prefill workitem 仍不可拆分。

## 关联

无(open PR 检索未发现重复修复)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
