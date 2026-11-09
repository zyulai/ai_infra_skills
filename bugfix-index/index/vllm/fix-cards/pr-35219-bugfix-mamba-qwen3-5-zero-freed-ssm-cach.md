# PR #35219: [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35219](https://github.com/vllm-project/vllm/pull/35219) |
| 作者 | vadiklyutiy |
| 合并日期 | 2026-03-10 |
| labels | bug, qwen, ready, ready-run-all-tests, v1 |
| 热度 | 👍 16 · 💬 42 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
混合模型复用的 KV block 残留 Mamba fp32 NaN 位模式污染注意力;新分配的 full-attention block 用单个 Triton kernel 先清零。

## 问题现象
混合模型(如 Qwen3.5-397B-A17B)在 attention(fp8/fp16)与 Mamba/SSM(fp32)层之间共享统一 block 池。先前被 Mamba(fp32 state)使用过的 block 重新分配给 dtype 更小的 attention 层时,残留的 fp32 位模式在新 dtype 下表现为 NaN/Inf。注意力 kernel(FlashAttn3、FlashInfer-TRTLLM 等)对未使用位置用乘零掩码,而 `0 * NaN = NaN` 清不掉 NaN;陈旧 NaN 随即传播到共享同一 KV-cache block 的**所有**请求,造成随时间累积的精度退化。复现:Qwen3.5-397B-A17B-FP8(TP=4、FlashInfer)连续跑 GSM8K,首次后精度显著下降。

## 根因
正文披露:block 池跨层复用且清零缺失——Mamba 用过的 fp32 block 被分配给 attention 层时不经清零,位模式在更窄 dtype 下解释为 NaN/Inf;attention kernel 的乘零掩码无法中和 NaN,污染沿共享 block 扩散。(flash-attention 侧问题见 Dao-AILab/flash-attention#1974,本 PR 是其 workaround。)

## 修复方案
仅对混合模型、仅对新分配的 `FullAttentionSpec` block、在使用前清零(prefix-cached 命中块不清零;Mamba/SSM 块不清零,因其每步全量覆写状态)。三部分:
1. Scheduler 侧:`SingleTypeKVCacheManager` 跟踪上次调度以来分配的 block ID(`allocate_new_blocks`/`allocate_new_computed_blocks` 中仅 FullAttentionSpec 追加),调度后经 `take_new_block_ids()` 排入 `SchedulerOutput.new_block_ids_to_zero`,由 `needs_kv_cache_zeroing`(= `has_mamba_layers`)门控。
2. Worker 侧:`GPUModelRunner._update_states()` 调 `_zero_block_ids()`,单个 Triton kernel `_zero_kv_blocks_kernel` 一次 launch 清零全部 KV-cache 段的对应内存。
3. 优化:`_init_kv_zero_meta()` 一次性预计算所有 KV 段的绝对字节地址(兼容 block-dim-0/1 布局、multi-buffer、虚拟块拆分);block ID 经预分配 pinned memory 异步 H2D 与 kernel launch 重叠;避免 Qwen3.5-379B 每层一次共 15 次 `index_fill_`。
关键 diff 摘录(vllm/v1/core/sched/output.py):
```python
+    # Block IDs freshly allocated from the pool during this scheduling step.
+    # The worker zeros the corresponding GPU memory before the blocks are used,
+    # preventing stale NaN/data from corrupting attention or SSM computation.
+    new_block_ids_to_zero: list[int] | None = None
```
CuMem 兼容:`_init_kv_zero_meta()` 在 `gpu_worker.py` 的 CuMem pool context **之外**调用,记账张量用标准 PyTorch allocator,sleep/wake 周期不被丢弃。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;受影响的是含 Mamba 层的混合模型(如 Qwen3.5)。
- 回归面:`vllm/v1/core/sched/`(scheduler/output/single_type_kv_cache_manager)、`kv_cache_manager.py`、`kv_cache_interface.py`(新增 `has_mamba_layers`/`needs_kv_cache_zeroing` property)、`gpu_model_runner.py`/`gpu_worker.py`、`v1/worker/utils.py`(新增 `KVBlockZeroer` 与 Triton kernel)、`attention/backend.py`(新增 `get_kv_cache_block_dim`)。非混合纯 attention 模型完全不受影响(门控短路)。开销:B200 实测 prefill ~170 μs(约 18ms step 的 1%)、decode ~15 μs(约 13ms step 的 0.1%);E2E 吞吐在 ±2% 噪声内。

## 关联
- https://github.com/vllm-project/vllm/issues/35138(所修 issue)
- https://github.com/Dao-AILab/flash-attention/issues/1974(上游 workaround)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
