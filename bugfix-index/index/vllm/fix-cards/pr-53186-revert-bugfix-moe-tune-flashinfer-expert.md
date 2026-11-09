# PR #53186: Revert "[Bugfix][MoE] Tune FlashInfer experts to scheduler token limit" (#52989)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53186](https://github.com/vllm-project/vllm/pull/53186) |
| 作者 | vllm-agent |
| 合并日期 | 2026-08-21 |
| labels | bug, ready, nvidia, quantization |
| 热度 | 👍 2 · 💬 3 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论

#52989 把 FlashInfer MoE autotune bucket 抬到 32768 导致 B200 PCP 评测 CUDA OOM,整体 revert 恢复 8192。

## 问题现象

合并 #52989(commit `bfb6c1349`)后的 nightly build #84887 首次把 `:nvidia: (B200) LM Eval PCP` 任务打红(此前四个 nightly 84473/84555/84687/84753 均为绿)。`evals/gsm8k/test_gsm8k_correctness.py::test_gsm8k_correctness[GLM-5.2-NVFP4-TP2-PCP2-EP]` 报 `Server exited unexpectedly.`,四个 worker 全部抛出:

```text
MemoryError: CUDA out of memory. Tried to allocate 12.22 GiB.
GPU 0 has a total capacity of 178.35 GiB of which 9.68 GiB is free.
  File ".../flashinfer_cutlass_moe.py", line 368, in apply
  File ".../flashinfer/fused_moe/core.py", line 636, in cutlass
  File "<unknown>", line 0, in FusedMoeRunner::getWorkspaceInfo(...)
```

而每 rank 占用更小的 `GLM-5.2-NVFP4-TP1-PCP4-EP` 配置仍通过(0.9325)。

## 根因

PR 正文披露:该 job 以 `max_num_batched_tokens=32768` 运行,#52989 加的 `tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config)` 使 `fi_moe_largest_bucket()` 返回 `max(32768 * dp_size, 8192)`,FlashInfer CUTLASS autotune bucket 从 **8192 跳到 32768`,`getWorkspaceInfo` 预留的 workspace 随之翻 4 倍,触发 OOM。`fi_moe_largest_bucket` 自己的 docstring 就警告"高估可能危险,会增加调优成本与内存使用"且其估算未建模 PCP——恰好正是出事的配置。

## 修复方案

对 #52989 做精确整体 revert(6 文件、1 插入、12 删除,无冲突):从 `flashinfer_cutlass_moe.py`、`trtllm_bf16_moe.py`、`trtllm_lora_moe.py`、`trtllm_mxint4_moe.py` 的 apply/invoke 调用中移除 `tune_max_num_tokens=fi_moe_largest_bucket(...)`,`flashinfer_trtllm_mxint4_moe` 恢复硬编码默认 `tune_max_num_tokens=8192`,并回滚 `tests/kernels/moe/test_flashinfer_moe.py` 中相应的参数与断言。

```diff
-            use_w4_group_scaling=use_w4_group_scaling,
-            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
+            use_w4_group_scaling=use_w4_group_scaling,
```

正文说明:这是 CI failure analyzer 自动生成的 **draft** revert;#52989 想解决的 autotune 覆盖问题本身是真实的,更窄的修复(对 tuned bucket 设上限或计入 PCP/EP 内存余量)更可取,revert 只是为了先让 B200 PCP 评测恢复绿色。

## 影响与回归风险

- 影响窗口:#52989 合并(commit `bfb6c1349`)至本 revert 之间的提交。
- 回归面:四个 FlashInfer/TRTLLM MoE experts 的 `tune_max_num_tokens` 传参与 mxint4 包装函数签名;revert 后 autotune bucket 回到默认 8192,#52989 原本想覆盖的调度 token 上限场景重新退回旧行为(等待更窄的后续修复)。

## 关联

- https://buildkite.com/vllm/ci/builds/84887
- https://github.com/vllm-project/vllm/pull/52989
- https://github.com/vllm-project/vllm/issues/51071

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
