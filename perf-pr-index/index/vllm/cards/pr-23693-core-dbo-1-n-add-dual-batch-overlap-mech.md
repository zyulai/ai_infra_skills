# PR #23693: [Core/DBO][1/N] Add Dual-Batch Overlap mechanism to VLLM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @SageMoore |
| 合并 | 2025-09-16 |
| 分类 | Overlap / Scheduler（tier: major） |
| Label | documentation, speculative-decoding, ready, v1 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "10 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/23693 |

## 一句话结论

新增 Dual-Batch Overlap（`--enable-microbatching` + 全 decode 批次），双线程双 stream 重叠 all-to-all 与 compute；PR 未给出定量性能数据。

## 动机

正文 Purpose：本 PR 为 VLLM 增加 Dual-Batch Overlap (DBO) 支持，当前仅在用户提供 `--enable-microbatching` 时启用，且仅当所有 DP groups 都在跑 full-decode batches 时使用。PR 支持在 full cudagraphs 下运行 DBO，正文称这对最小化 CPU overhead、从该特性获得性能至关重要。正文未给出改动前的定量瓶颈数据。

## 优化点

高层做法：把 batch 拆成两个 microbatches，用两个线程与两个 cuda streams（一个 communication、一个 computation），让一个 microbatch 的 dispatch/combine all-to-all kernels 与另一个 microbatch 的 compute kernels 重叠。启用时 GPUModelRunner 把 batch 切成两个 token_slices，在 `_prepare_inputs` 中传给 attention metadata builders，为每个 microbatch 生成独立 metadata；实际执行时起两个 microbatching 线程，通过 `UBatchContext` 互通，各自用对应 metadata 跑 `self.model`。为产生重叠，在 all-to-all kernels 中插入 "yield"：`yield_and_switch_from_compute_to_comm` 把 CPU 从线程 A 让给线程 B，线程 A 恢复后切到 communication stream 派发 kernel；`yield_and_switch_from_comm_to_compute` 反向从 comm 切回 compute。GPU 与 CPU events 用于同步，正文强调必须保证同一时刻只有一个 microbatching 线程运行（另一个等 event），且两个 microbatch 的 yield 次数必须完全相同。

## 关键改动

```text
vllm/config/parallel.py: 新增 enable_dbo 与 dbo_decode_token_threshold（默认 32）配置
vllm/config/__init__.py: 断言 enable_dbo 时 VLLM_ALL2ALL_BACKEND 必须为 deepep_low_latency
vllm/forward_context.py: 新增 should_ubatch_across_dp，决定各 DP rank 是否一起 microbatch 并做 padding
vllm/v1/attention/backends/utils.py 与 tests/v1/attention/test_attention_splitting.py: UBatchSlice 切片/attention metadata 拆分及 UbatchSlice→UBatchSlice 重命名
vllm/model_executor/layers/fused_moe/*prepare_finalize.py 等: all-to-all / prepare-finalize 路径插入 yield 切换（pack 丢弃 10 个文件）
```

```diff
+    enable_dbo: bool = False
+    """Enable microbatching for the model executor."""
+
+    dbo_decode_token_threshold: int = 32
+    """The threshold for microbatching. If the number of tokens in the
+    request is greater than this threshold, microbatching will be used.
+    Otherwise, the request will be processed in a single batch."""
```

## 性能声明

PR 正文未给出定量性能数据（吞吐/延迟数字均未给出，perf claims 段标注 "(none detected by regex)"）。正文仅定性说明在 full cudagraphs 下运行 DBO "is essential for minimizing the CPU overhead and getting performance from this feature"。正文给出的 Test Plan 为 lm_eval 正确性与多节点 R1 smoke 验证，不构成性能测试条件；PR 未注明测试条件。

## 验证与回归风险

验证（Test Plan）：用 `deepseek-ai/DeepSeek-V2-Lite` 跑 lm_eval，另在多节点 R1 上多次运行确认 lm_eval 输出合理。gsm8k exact_match（5-shot，flexible/strict）：Non-DBO Eager 0.3567/0.3533（±0.0277/0.0276）、Non-DBO Default 0.3700/0.3667（±0.0279/0.0279）、DBO Eager（`--enable-microbatching --microbatching-token-threshold 4`）0.3800/0.3767（±0.0281/0.0280）、DBO Full cudagraphs（`cudagraph_mode full_decode_only`）0.3733/0.3700（±0.0280/0.0279）；命令均带 `VLLM_ALL2ALL_BACKEND=deepep_low_latency`、`--data-parallel-size 2 --enable-expert-parallel`。可能影响路径：仅在 `--enable-microbatching` 且全 DP groups 为 full-decode、all2all backend 为 `deepep_low_latency` 时启用；正文强调双线程同步与 yield 次数必须严格一致，属潜在正确性风险；diff 另从 `all2all.py` 的 `get_handle` 中移除了 `handle.set_num_sms(self.num_sms)` 调用。PR 未注明更多回归风险。

## 关联

无（标题含 [1/N] 表明是系列首篇，但 PR 正文与 diff 未给出同系列其他 PR 编号或明确线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
