# PR #57779: [XPU][UT]Bugfix when the process can't see all the world_size meet accuracy issue.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57779](https://github.com/vllm-project/vllm/pull/57779) |
| 作者 | yisustc |
| 合并日期 | 2026-09-20 |
| labels | bug, intel-gpu |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

XPU 上进程看不到全部 world_size 设备时 uneven reduce_scatter 结果错误,改为 pad 到 max_size 走 reduce_scatter_tensor。

## 问题现象

当 DP 进程看不到完整 world_size 的设备时出现精度问题。复现(外部 LB DP 单测):

```bash
NCCL_CUMEM_HOST_ENABLE=0 TP_SIZE=1 DP_SIZE=2 pytest -sv v1/distributed/test_external_lb_dp.py::test_external_lb_completion_streaming[api-1-standard-ibm-research/PowerMoE-3b]
```

断言失败 `AssertionError: Streamed output should match non-streamed output.`,流式输出与非流式输出不一致:`'PictureConti...tinueContinue'` 与 `'PictureıFallıdw'`(模型输出乱码/重复,如 `PictureContinueContinueContinueContinue`),即 reduce_scatterv 结果数值已损坏。

## 根因

正文未披露详细根因,只指出场景是"进程看不到全部 world_size"(如 external-LB DP 单测)时 uneven reduce_scatter 出精度问题。(由 diff 推断)`xpu_communicator.reduce_scatterv` 中,各 rank 输入尺寸不均时走 `dist.reduce_scatter(output, input_splits)`(list 分组版);当进程无法看到组内所有设备(`torch.xpu.device_count() < world_size`)时该路径不被支持,结果不正确,新代码注释也明确写明 "uneven reduce_scatter is not supported when the process cannot see every device in the group (e.g. external-LBDP)"。

## 修复方案

改 `vllm/distributed/device_communicators/xpu_communicator.py` 的 `reduce_scatterv`,分三路处理:

- `sizes` 为 None 或各 rank 尺寸相同:直接 `dist.reduce_scatter_tensor`(原有等尺寸路径)。
- `torch.xpu.device_count() < self.world_size`(进程看不到全部设备):把每个 rank 的 split 按零 pad 到 `max(sizes)`,拼成 `(world_size * max_size, ...)` 后走 `reduce_scatter_tensor`,再切回 `output[:chunk_size]`。
- 其余(能看到全部设备且尺寸不均):保留原 `dist.reduce_scatter(output, input_splits)`。

```diff
+        elif torch.xpu.device_count() < self.world_size:
+            max_size = max(sizes)
+            padded = input_tensor.new_zeros(
+                (world_size * max_size,) + input_tensor.shape[1:]
+            )
+            for rank, split in enumerate(input_tensor.split(sizes, dim=0)):
+                padded[rank * max_size : rank * max_size + sizes[rank]] = split
+            padded_output = input_tensor.new_empty((max_size,) + input_tensor.shape[1:])
+            dist.reduce_scatter_tensor(padded_output, padded, group=self.device_group)
+            output = padded_output[:chunk_size]
```

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 XPU device communicator 的 `reduce_scatterv`;等尺寸路径与"可见全部设备的不均匀路径"行为不变,新增分支只覆盖"进程看不到全部设备"的场景(external-LB DP 等),额外付出一次 pad 拷贝的内存与带宽开销。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
