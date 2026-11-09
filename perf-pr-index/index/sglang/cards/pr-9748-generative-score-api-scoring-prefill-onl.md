# PR #9748: [Generative Score API] Scoring(Prefill-only) optimizations.

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @sundar24295s |
| 合并 | 2025-09-13 |
| 分类 | Attention / Kernel (tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/9748 |

## 一句话结论

Score API 的 prefill-only 打分请求跳过输入 token logprob 与 sampling,并把逐条 gather 合成单次批量 GPU gather,H100 上 P99 延迟 6220→454 ms(约 13.7x)。

## 动机

Score API(#8840 的后续)只需要完整 prompt 之后的 next-token 分布(如 P(Yes|full_prompt)、P(No|full_prompt)),既不需要 prompt 内部逐 token 的 logprob,也不需要采样。此前路径却照常计算输入 token logprob 并执行完整 sampling 步骤;profiling 显示 baseline 中 sampling 与 logits 提取引发 memsync,阻塞下一 batch 调度,高负载下批间 GPU kernel launch 出现明显空隙。

## 优化点

三个层面。其一,对 prefill-only 请求(`max_new_tokens == 0`,新增 `Req.is_prefill_only` 属性)把 `logprob_start_len` 设为 `len(origin_input_ids)`,并在 `prepare_for_extend` 中令 `extend_logprob_start_len = extend_input_len`,完全跳过输入 token 的 logprob 计算。其二,新增 `Sampler.compute_logprobs_only()` 与 `ModelRunner.compute_logprobs_only()`:skip_sample 时仍对 requested token 计算 log_softmax 后的定向 logprob,但不做采样;overlap 路径给 `forward_batch_generation` 加 `skip_sample` 参数,prefill-only 用 CPU 上的 dummy token id 填充 future map。其三,新增 `get_token_ids_logprobs_batch_optimized()`:整个 batch 只做一次向量化 `logprobs[row_indices, col_indices]` gather,再按各请求长度 `split_with_sizes` 拆回;`delay_cpu_copy=True` 可把 `.tolist()` 推迟到结果处理阶段,与下一批计算重叠。

## 关键改动

```text
python/sglang/srt/layers/sampler.py: compute_logprobs_only() 与批量向量化 gather
python/sglang/srt/managers/schedule_batch.py: is_prefill_only 判定与 extend_logprob_start_len 跳过逻辑
python/sglang/srt/managers/tp_worker.py / tp_worker_overlap_thread.py: skip_sample 路径与延迟 CPU 拷贝
python/sglang/srt/managers/scheduler.py: prefill-only 的 logprob_start_len 默认值调整
```

```python
+                if self.is_prefill_only and req.logprob_start_len == len(
+                    req.origin_input_ids
+                ):
+                    # Skip ALL input logprobs
+                    req.extend_logprob_start_len = req.extend_input_len
```

## 性能声明

Qwen3-0.6B,H100(CUDA 12.8),300-token prompt,每请求 10 items,Poisson 分布,120 s,server 启动含 `--enable-torch-compile --disable-radix-cache --disable-cuda-graph` 等参数:

- QPS 100:P99 延迟 6220 ms → 454 ms(约 13.7x,降 92.7%)。
- P99 阈值 500 ms 下:吞吐 800 → 1000 items/s/H100(约 +25%)。
- 各目标 RPS(600–1200)的 P99 对比:800 items/s 时 413.14→227.20 ms,900 时 1200.72→302.39 ms,1100 时 8694.97→1459.81 ms。
- 精度:同一 score 请求改前改后 scores 数组逐值相同。

## 验证与回归风险

- PR 用改前/改后 curl 返回相同 scores 作为准确性证据;未新增单元测试(checklist 中 Add unit tests 未勾选)。
- 回归面:`logprob_start_len` 默认值语义变化只作用于 prefill-only;`next_token_token_ids_logprobs_val` 类型放宽为可含 GPU tensor,`tp_worker_overlap_thread` 对 CPU 上的 next_token_ids 跳过重复拷贝;非 prefill-only 路径走原 `min(...)` 逻辑不变。测试条件中禁用 CUDA graph/radix cache,与常规生成服务配置不同,数字不宜直接外推。

## 关联

- 前序:#8840(generative score API 引入)。
- 无同区域其他 PR 编号被正文提及。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
