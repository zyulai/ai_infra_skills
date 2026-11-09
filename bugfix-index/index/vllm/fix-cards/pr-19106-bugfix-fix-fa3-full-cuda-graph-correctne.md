# PR #19106: [Bugfix] Fix FA3 full cuda graph correctness

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19106](https://github.com/vllm-project/vllm/pull/19106) |
| 作者 | WoosukKwon |
| 合并日期 | 2025-06-04 |
| labels | ready, ci/build, v1 |
| 热度 | 👍 6 · 💬 4 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
FA3 配合 full cuda graph 输出不正确;强制 AOT 调度并把 scheduler_metadata 固定到持久缓冲后清零补齐。

## 问题现象
使用 full cuda graph 时出现正确性问题(正文未附具体日志或评测数字);同时把 `test_full_cudagraph.py` 加入 CI(Qwen2-1.5B-Instruct,仅 Hopper (9,0) GPU 运行,测试的 gpu_memory_utilization 由 0.2/0.5 上调至 0.3/0.6)。

## 根因
正文未披露根因机制,只说修复 full cuda graph 的正确性问题。从 diff 看:full cuda graph 下 FA3 曾禁用 AOT 调度,且 `build()` 产出的 scheduler_metadata 是临时张量,图重放时部分 thread block 会读到无效 scheduler metadata 并覆盖输出缓冲(diff 中 NOTE 注释原文),故需持久缓冲并在拷入后清零剩余部分;此外 `_dummy_run` 图捕获时 `seq_lens` 未按 `max_model_len` 填充(均由 diff 推断)。

## 修复方案
`vllm/v1/attention/backends/flash_attn.py`:FA3 下 `aot_schedule` 恒为 True;`full_cuda_graph` 且无 AOT 调度时直接抛 ValueError("Full CUDA graph mode requires AOT scheduling, which requires FlashAttention 3.");预分配 `max_num_reqs + 1` 的 `scheduler_metadata` 缓冲,`build()` 时拷入本次内容并清零剩余;`max_seq_len` 转为 `int`。`vllm/v1/worker/gpu_model_runner.py` `_dummy_run`:图捕获时把 `seq_lens_np[:num_reqs]` 填为 `max_model_len`、其余置 0。CI 增加 `compile/piecewise/test_full_cudagraph.py`。

```diff
+        if self.use_full_cuda_graph:
+            assert scheduler_metadata is not None
+            n = scheduler_metadata.shape[0]
+            self.scheduler_metadata[:n].copy_(scheduler_metadata,
+                                              non_blocking=True)
+            # NOTE(woosuk): We should zero out the rest of the scheduler
+            # metadata to guarantee the correctness. Otherwise, some thread
+            # blocks may use the invalid scheduler metadata and overwrite the
+            # output buffer.
+            self.scheduler_metadata[n:] = 0
+            scheduler_metadata = self.scheduler_metadata[:n]
```

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:FlashAttention3 后端元数据构建路径;非 FA3 后端 + full_cuda_graph 组合现在启动即报错(不再静默禁用 AOT);full cuda graph 捕获的 dummy run 语义变化;CI 流水线新增步骤。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
