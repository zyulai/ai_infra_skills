# PR #2051: Fix illegal memory access in overlap mode & Use more fused triton kernels for building meta data

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2051](https://github.com/sgl-project/sglang/pull/2051) |
| 作者 | merrymercy |
| 合并日期 | 2024-11-17 |
| labels | (none) |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
overlap 模式下 CUDA illegal memory access,在 ScheduleBatch.copy() 前加流同步修复;同时用融合 Triton kernel 构建 prefill metadata 并简化 copy queue。

## 问题现象
正文仅说明 overlap 模式(overlap model)下存在 CUDA illegal memory access,未描述具体复现步骤、报错日志或影响面。

## 根因
正文未披露根因。(由 diff 推断)`ScheduleBatch.copy()` 的 diff 中新增了明确注释:"We need a stream synchronization here. Otherwise, there will be cuda illegal memory access errors."——即 overlap 调度线程在复制 batch 时,`seq_lens` 等设备张量上的异步操作尚未完成(缺少流同步),导致后续读取触发非法内存访问;修复即通过 `_ = self.seq_lens[0].item()` 强制一次 GPU→CPU 同步。

## 修复方案
三部分改动:

1. **修 IMA**:`python/sglang/srt/managers/schedule_batch.py` 的 `copy()` 开头加 `_ = self.seq_lens[0].item()` 引入同步。
2. **融合 Triton kernel 构建 prefill metadata**:
   - `schedule_batch.py` 新增 `write_req_to_token_pool_triton`,替换 `prepare_for_extend` 中逐请求的 `req_to_token_pool.write((req.req_pool_idx, slice(pre_len, seq_len)), ...)` Python 写入;
   - `forward_batch_info.py` 新增 `compute_position_triton`/`compute_position_kernel`,替换 `init_new` 里 `torch.concat([torch.arange...])` 的 positions 计算并同时产出 `extend_start_loc`。
   - `flashinfer_backend.py` 的 `create_flashinfer_kv_indices_triton` 同步重写索引计算,参数 `max_context_len` 改为 `req_to_token_ptr_stride`,并移除实例上缓存的 `max_context_len`。
3. **简化 overlap copy queue**:`tp_worker_overlap_thread.py` 删除独立 `copy_thread`/`copy_queue`,`copy_event`(改为非 blocking)连同结果直接进 `output_queue`,由 `resulve_batch_result` 等待 `copy_event` 并做 `.tolist()` 转换;`launch_event.wait()` 改为无条件等待。

另:`logits_processor.py` 仅在 `return_logprob` 且 extend 模式时计算 `extend_logprob_pruned_lens_cpu`(改用 `extend_seq_lens_cpu`);`sampling_batch_info.py` 的 `is_all_greedy` 由 `top_ks.max().item() <= 1` 改为 `all(r.sampling_params.top_k <= 1 for r in reqs)`;`bench_offline_throughput.py` 加 `--do-not-exit` 便于 nsys profile。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:overlap 调度全路径(新增同步点,理论上有轻微吞吐代价);prefill 的 req_to_token 写入与 positions 计算(改由 Triton kernel 承担,语义需与原 Python/PyTorch 实现等价);flashinfer 后端的 kv_indices 构建;logprob 剪枝长度计算(非 logprob 或非 extend 路径现在跳过该计算);`is_all_greedy` 判定改为按原始采样参数逐一判断。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
