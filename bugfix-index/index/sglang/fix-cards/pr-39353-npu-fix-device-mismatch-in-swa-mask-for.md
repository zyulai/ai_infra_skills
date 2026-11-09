# PR #39353: [NPU] Fix device mismatch in SWA mask for DSpark verify graph capture

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39353](https://github.com/sgl-project/sglang/pull/39353) |
| 作者 | iridiumine |
| 合并日期 | 2026-09-14 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DSpark verify 走 decode_cuda_graph_runner 时传入 CPU 上的 `seq_lens`,与 NPU 上的 `swa_indices` 比较触发设备不匹配崩溃;补一行 `.to(self.device)` 修复。

## 问题现象
DSpark verify 复用 `DFlashVerifyInput`,使 `_is_dflash_verify()` 在 NPU attention backend 的 SWA mask 更新中把它路由进 `seq_lens_cpu` 分支。与 DFlash 不同(DFlash 的 target verify graph 走 NPU graph runner、`seq_lens` 在设备上),DSpark 的 verify graph 走 `decode_cuda_graph_runner`,传入的是 CPU 张量(capture 时为 `forward_batch.seq_lens.cpu()`,replay 时为 `buffers.seq_lens_cpu`)。与设备上的 `swa_indices` 缓冲比较时抛出:
```
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor, please check whether the input tensor device is correct.
```
崩溃发生在 target model 的 graph capture 阶段,导致 DSpark 无法在带 hybrid SWA 层的 Ascend NPU 上运行。

## 根因
正文披露:`_apply_cuda_graph_metadata` 的 `seq_lens_cpu` 分支假设该张量已在 NPU 设备上,直接与设备上的 `swa_indices` 比较;但 DSpark verify 经 `decode_cuda_graph_runner` 传入的 `seq_lens_cpu` 是 CPU 张量,跨设备比较触发 RuntimeError。附带影响(正文一并披露):由于 `seq_lens_cpu` 为 prefix + block_size,而 prefix-only 的 `seq_lens` 会把 draft KV 位置错误掩蔽,修好后 DSpark 同时获得正确的 SWA mask 窗口。

## 修复方案
`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` 的 `_apply_cuda_graph_metadata`,一行修复——在 SWA mask 比较前把 `seq_lens_int` 搬到 backend 设备:
```diff
-                seq_lens_int = seq_lens_cpu[:bs].int()
+                # seq_lens_cpu may be a CPU tensor; swa_indices is on-device.
+                seq_lens_int = seq_lens_cpu[:bs].int().to(self.device)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 NPU attention backend 的 cuda graph 元数据准备路径;对 DFlash 路径为 no-op(其 `seq_lens_cpu` 本就在设备上,`.to(self.device)` 不产生拷贝);新增的传输是 replay 准备阶段(捕获图之外)的主机侧 `bs × 4` 字节拷贝(约微秒级),正文评估精度与速度均无影响;顺带修正 DSpark 的 SWA 掩码窗口语义(draft KV 不再被 prefix-only `seq_lens` 掩蔽)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
