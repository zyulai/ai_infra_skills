# PR #50183: [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50183](https://github.com/vllm-project/vllm/pull/50183) |
| 作者 | gabriel-peracio |
| 合并日期 | 2026-08-06 |
| labels | bug, mrv2, ready, speculative-decoding, v1 |
| 热度 | 👍 2 · 💬 7 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
全 NaN 块上 tl.argmax 返回越界块索引,拒绝采样器 OOB 读致 IMA;argmax 前 NaN 映射 -inf 修复。

## 问题现象
投机解码预热阶段发生 illegal memory access(IMA):在 8xB300、TP=8 上以 DSpark 投机解码服务 Kimi-K3 时观察到。触发条件是目标 logits 全为 NaN,使(填充后的)块维度内所有局部最大值均为 NaN。作者另在 RTX 4090 上用独立 Triton 复现脚本确认:全 NaN 输入下未打补丁的 `tl.argmax` 返回 `idx == num_blocks`(越界;NUM_BLOCKS=4、PAD=8 时输出 8),打补丁后索引落在 [0, 3] 内。

## 根因
正文披露:`tl.argmax` 在全 NaN 块上不保证返回范围内的结果,会给出指向 `num_blocks - 1` 之后填充区域的越界块索引;该索引随即被用于从 local-argmax 张量加载,造成越界读,垃圾索引继续传播进后续 gather/store,最终表现为拒绝采样器中的 IMA。两个内核受影响:`_compute_global_target_argmax`(L111 的 `tl.argmax(local_max)` 结果在 L113 索引 `target_local_argmax_ptr`)与 `_insert_resampled_kernel`(L852 的 `tl.argmax(resampled_local_max)` 结果在 L854 索引 `resampled_local_argmax_ptr`)。

## 修复方案
改 `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` 两个内核,在 `tl.argmax` 之前用 `tl.where` 把 NaN 映射为 `-inf`(`local_max != local_max` 判 NaN):

```diff
+    # NaN max values (from NaN target logits) make tl.argmax return an
+    # out-of-range block index (into the padded region), causing an OOB read
+    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
+    resampled_local_max = tl.where(
+        resampled_local_max != resampled_local_max,
+        float("-inf"),
+        resampled_local_max,
+    )
```

正文还提到对结果块索引用 `tl.minimum` 钳制到 `num_blocks - 1` 作防御边界,但 pack 提供的 diff 摘录中未见对应改动行,以实际代码为准。新增回归测试 `test_all_nan_target_logits_in_range`(temperature 0.0/1.0,vocab 20000 产生 3 块填到 4 的非 2 次幂块数),断言内核无错完成且输出 token id 在范围内。硬件验证:8xB300 TP=8 Kimi-K3 DSpark 预热通过(此前 IMA),输出连贯,acceptance length 2.5,conc=1 吞吐 110-116 tok/s(较无投机 +58%)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;受影响的是 rejection sampler 使用这两个 Triton 内核的版本,需目标 logits 出现 NaN 才触发。
- 回归面:两个内核各加一处 NaN→-inf 映射,非 NaN 输入行为不变(正文明确);非 2 次幂块数(存在填充区)是最先暴露的场景。全 NaN logits 本身意味着模型已输出异常,本修复把故障从进程级 IMA 降为该请求输出异常;若下游依赖「采样器遇到 NaN 即崩溃」的假设则需留意,正文未列举此类依赖。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
