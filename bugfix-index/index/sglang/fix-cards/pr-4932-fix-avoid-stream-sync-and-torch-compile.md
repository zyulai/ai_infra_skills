# PR #4932: [Fix] avoid stream sync and torch compile in prefill for fa3 backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4932](https://github.com/sgl-project/sglang/pull/4932) |
| 作者 | Fridge003 |
| 合并日期 | 2025-03-30 |
| labels | (none) |
| 热度 | 👍 5 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
fa3 prefill 每批有 ~100ms stream sync 气泡、启动多 ~4s 编译开销;改用 seq_lens_cpu 并移除 clamp_position 的 torch.compile。

## 问题现象
profiling fa3 backend 发现两处开销(正文):(a) `init_forward_metadata` 中 `.item()` 操作造成 cuda stream 同步气泡,每个 prefill/extend batch 都发生,约 100ms;(b) torch compiler 编译 `ForwardBatch.init_new` 调用的 `clamp_position`,只在启动时发生一次,但耗时约 4s。基准(8*H200 + DeepSeek-V3-0324,bench_serving 480 prompts、random-input 4000):请求吞吐 2.43→3.06 req/s,Mean TTFT 54368.99→35401.94 ms,Mean E2E 130081.39→93933.39 ms;GSM8K 0.941、MMLU 0.876。

## 根因
正文披露:(a) 同步气泡来自 GPU 上的 `seqlens_in_batch.max().item()`;(b) 启动开销来自 `clamp_position` 被 `@torch.compile` 编译。

## 修复方案
1. 引入 `seq_lens_cpu`:按正文,worker batch 在 extend 模式也把该张量拷贝到 CPU。diff 上 `schedule_batch.py` 的 `get_model_worker_batch` 不再仅 decode 时构造,凡 fa3 / flashinfer_mla / flashmla 后端统一构造 `seq_lens_cpu = self.seq_lens.cpu()`;fa3 `init_forward_metadata` 改用 `forward_batch.seq_lens_cpu.max().item()`;`decode_seq_lens`/`decode_seq_lens_cpu` 统一更名为 `seq_lens_cpu`(flashinfer_mla、flashmla、cuda_graph_runner、eagle_draft_cuda_graph_runner 同步)。
2. `forward_batch_info.py` 删除被 torch.compile 的 `clamp_position`,decode 分支改为内联 `torch.clamp((batch.seq_lens - 1), min=0).to(torch.int64)`。

```diff
-        metadata.max_seq_len_k = seqlens_in_batch.max().item()
+        metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
```

```diff
-@torch.compile(dynamic=True, backend=get_compiler_backend())
-def clamp_position(seq_lens):
-    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)
```

## 影响与回归风险
- 影响窗口:正文未给出版本/提交范围。
- 回归面:按 diff,`ModelWorkerBatch`/`ForwardBatch` 字段更名(`decode_seq_lens`→`seq_lens_cpu`、`decode_seq_lens_cpu`→`seq_lens_cpu`)波及 fa3、flashinfer_mla、flashmla 三个后端与 cuda graph runner、EAGLE draft runner 的批处理元数据构造;启用上述后端时 prefill 也多一次 seq_lens 的 CPU 拷贝;`clamp_position` 删除改变 decode 模式 positions 的计算路径(改为未编译版本)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
