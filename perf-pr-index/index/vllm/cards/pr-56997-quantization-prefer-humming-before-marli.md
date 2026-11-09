# PR #56997: [Quantization] Prefer Humming before Marlin backends on SM90

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2026-10-06 |
| 分类 | Quantization \| tier: major |
| Label | performance, ready, nvidia, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/56997 |

## 一句话结论

SM90 上把 Humming kernel 优先级提到 Marlin 之前（PR 正文未给定量数据）。

## 动机

正文说明：Humming 在 SM90 支持 WGMMA，性能显著优于 Marlin，并且支持 MXFP4 × FP8 等更高效路径，因此是时候更新这一优先级。

## 优化点

在线性 kernel 选择优先级上把 Humming 排到 Marlin 之前。`vllm/model_executor/layers/quantization/utils/humming/priority.py` 新增重排逻辑：取 kernels 名称列表，找到含 humming 与含 marlin 的索引，若两者都存在且 humming 位于 marlin 之后，则把 humming 弹出并插入到 marlin 位置，其余情况原样返回。重排在 kernels 列表副本上进行，不改变原列表；当 Humming 已位于 Marlin 之前或任一缺失时直接返回原列表。多个 `fused_moe/oracle/*.py` 与 `linear/__init__.py`、相关 tests 同步接入。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/humming/priority.py: 新增 kernel 优先级重排（Humming 前置于 Marlin）
vllm/model_executor/kernels/linear/__init__.py、mixed_precision/humming.py: 线性 kernel 选择接入
vllm/model_executor/layers/fused_moe/oracle/{fp8,int_wna16,mxfp4,mxfp8,nvfp4}.py: MoE oracle 优先级同步
tests/kernels/quantization/test_scaled_mm_kernel_selection.py、test_modelopt.py、test_online.py: 选择顺序测试
```

```python
+    humming = next((i for i, name in enumerate(names) if "humming" in name), None)
+    marlin = next((i for i, name in enumerate(names) if "marlin" in name), None)
+    if humming is None or marlin is None or humming < marlin:
+        return kernels
+    kernels = kernels.copy()
+    kernels.insert(marlin, kernels.pop(humming))
+    return kernels
```

## 性能声明

PR 正文未给出定量性能数据（仅定性说明 Humming 在 SM90 因支持 WGMMA 与 MXFP4×FP8 而「delivers significantly better performance than Marlin」）。PR 未注明测试条件。

## 验证与回归风险

正文未给出测试计划或测试结果；diff 中含 `tests/kernels/quantization/test_scaled_mm_kernel_selection.py`、`tests/quantization/test_modelopt.py`、`tests/quantization/test_online.py` 等选择顺序相关测试。回归风险：改动影响 SM90 上量化线性/MoE kernel 的后端选择顺序，涉及 fp8、int_wna16、mxfp4、mxfp8、nvfp4 等 oracle；正文未声明对其他架构的影响。

## 关联

无（正文未给出任何 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
