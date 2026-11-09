# PR #43646: [XPU] Fix fused MoE LoRA kernel crash on XPU by using platform-agnos num_compute_units

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43646](https://github.com/vllm-project/vllm/pull/43646) |
| 作者 | chaojun-zhang |
| 合并日期 | 2026-05-26 |
| labels | intel-gpu, ready |
| 热度 | 👍 4 · 💬 0 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
XPU 上运行 fused MoE LoRA kernel 直接调用 `torch.cuda.get_device_properties()` 触发 "Torch not compiled with CUDA enabled" 断言崩溃;换成平台无关的 `current_platform.num_compute_units()` 修复。

## 问题现象
在 XPU(Intel GPU)上运行 fused MoE LoRA kernel 时崩溃,报错为 `AssertionError: Torch not compiled with CUDA enabled`。触发场景即在任何未编译 CUDA 支持的构建上调用 fused MoE LoRA 相关路径(网格/小批量启发式需要读取设备 SM 数量)。正文未提供完整堆栈或版本号。

## 根因
PR 正文披露:`_fused_moe_lora_grid` 与 `_run_fused_moe_lora_small_batch` 内直接调用 `torch.cuda.get_device_properties()`(注:diff 中实际改动的两处调用点位于 `_run_fused_moe_lora_one_shot` 与 `_run_fused_moe_lora_small_batch`);在 XPU 平台 torch 未启用 CUDA,该调用直接断言失败导致崩溃。

## 修复方案
改 `vllm/lora/ops/triton_ops/fused_moe_lora_op.py`:新增 `from vllm.platforms import current_platform`,把两处 `sm_count = torch.cuda.get_device_properties(device).multi_processor_count` 替换为 `sm_count = current_platform.num_compute_units(device.index)`——该接口已处理平台分派,各平台返回各自的计算单元数,不再触碰 CUDA API。

```diff
-    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
+    sm_count = current_platform.num_compute_units(device.index)
```

(两处调用点相同,均替换。)

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:fused MoE LoRA triton 算子的网格启发式(NPID_FACTOR、small-batch chunk 划分)依赖 `sm_count`;CUDA 平台行为取决于 `num_compute_units` 与 `multi_processor_count` 的数值等价性(diff 未展示该接口实现),XPU 平台从崩溃变为可运行。逻辑其余部分未改动。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
