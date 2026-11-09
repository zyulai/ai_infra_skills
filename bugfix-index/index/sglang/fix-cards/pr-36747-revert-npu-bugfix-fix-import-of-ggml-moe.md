# PR #36747: Revert "[NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#36747](https://github.com/sgl-project/sglang/pull/36747) |
| 作者 | Fridge003 |
| 合并日期 | 2026-08-27 |
| labels | hicache |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回退 #36640:撤销 NPU 上 expert_pack 条件导入与 MLA HiCache 备份跳过 data_ptrs 的两处"修复",恢复通用逻辑。

## 问题现象
正文仅一句 "Reverts sgl-project/sglang#36640",未描述具体现象。被回退的原 PR 标题为 "[NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs",本 PR 将其整体撤销;具体症状与复现条件正文未给出。

## 根因
正文未披露根因,仅说明是对 #36640 的 revert。(由 diff 推断)机制,回退恢复了两处原逻辑:
1. `mxfp4_flashinfer_trtllm_moe.py` 的 `maybe_fuse_routed_scale_and_shared_add` 中,`ExpertPackMoEMethod` 的导入恢复为无条件(函数顶部直接 `from ... expert_pack import ExpertPackMoEMethod`),isinstance 判断恢复为固定的四元组;#36640 曾把该导入改为 `if not is_npu()` 才执行(以此规避 NPU 上的 ggml_moe_a8_vec 导入问题)。
2. `mem_cache/pool_host/mla.py` 的 `backup_from_device_all_layer` 中,恢复无条件调用 `self._resolve_device_transfer_buffers(device_pool)`;#36640 曾为 `io_backend == "kernel_ascend"`(NPU)跳过解析 `device_data_ptrs, device_kv_buffers`(NPU 池使用连续多层张量、不构建 CUDA 风格 data_ptrs 数组)。

## 修复方案
整体 revert #36640,两个文件:
- `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py`:删除 `is_npu` 导入,`ExpertPackMoEMethod` 无条件导入并参与 isinstance 判断。
- `python/sglang/srt/mem_cache/pool_host/mla.py`:删除 `kernel_ascend` 分支特判。

关键 diff(mla.py):
```diff
-        if io_backend == "kernel_ascend":
-            # NPU pools use contiguous multi-layer tensors and intentionally do
-            # not build the CUDA-style data_ptrs array.
-            device_data_ptrs, device_kv_buffers = None, None
-        else:
-            device_data_ptrs, device_kv_buffers = self._resolve_device_transfer_buffers(
-                device_pool
-            )
+        device_data_ptrs, device_kv_buffers = self._resolve_device_transfer_buffers(
+            device_pool
+        )
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间,即 #36640 合入后至本回退落地之间。
- 回归面:NPU 相关两条路径——① mxfp4 FlashInfer TRTLLM MoE 的 fused scale/add 判断(NPU 上重新引入 ExpertPackMoEMethod 导入链,若 ggml_moe_a8_vec 导入问题仍在,相关模型加载会受影响);② NPU MLA HiCache 的设备备份路径(恢复解析 data_ptrs/kv buffers,若 NPU 池确实缺失这些指针,备份可能重新出错)。本 PR 标注了 hicache label。

## 关联
- [sgl-project/sglang#36640](https://github.com/sgl-project/sglang/pull/36640)(正文注明的被 revert 对象)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
