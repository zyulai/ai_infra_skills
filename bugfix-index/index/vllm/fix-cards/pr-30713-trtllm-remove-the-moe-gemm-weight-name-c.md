# PR #30713: [TRTLLM] Remove the MoE GEMM weight name change

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30713](https://github.com/vllm-project/vllm/pull/30713) |
| 作者 | minosfuture |
| 合并日期 | 2025-12-16 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | hang |

## 一句话结论
TRT-LLM MoE 的 GEMM 权重改名无功能意义却破坏依赖标准权重名的 offload 功能,恢复 w13/w2 命名并保留权重修复。

## 问题现象
TRT-LLM MoE FP4 路径中,权重 offloading 功能(依赖模型代码给出待 offload 权重列表)被破坏。正文说明该改名「doesn't serve any functionality purpose」,即单纯把 `w13_weight`/`w2_weight` 等改名为 `gemm1_weights_fp4_shuffled` 等新名字,导致按标准权重名工作的下游机制失效。

## 根因
正文披露根因:此前引入的 GEMM weight/scale 参数名变更(`w13_weight`→`gemm1_weights_fp4_shuffled`、`w2_weight`→`gemm2_weights_fp4_shuffled`、`w13_weight_scale`→`gemm1_scales_fp4_shuffled`、`w2_weight_scale`→`gemm2_scales_fp4_shuffled`,并 `del` 掉原名字段)没有任何功能上的必要,却破坏了 PR #29941 的 weight offloading 功能——该功能依赖模型代码指定一个按名字引用的权重列表来做 offload,改名后列表里的标准名字找不到了。

## 修复方案
撤销权重改名,恢复标准命名(三个文件):
1. `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` 与 `vllm/model_executor/layers/quantization/modelopt.py` 的 `process_weights_after_loading`:TRT-LLM shuffle 分支重新把 shuffled 结果写回 `layer.w13_weight`、`layer.w2_weight`、`layer.w13_weight_scale`、`layer.w2_weight_scale`,并删除「Clean up weights that won't be used by TRT-LLM」的四个 `del` 语句(不再删除原权重)。
2. `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` 的 `flashinfer_trtllm_fp4_moe` 与 `flashinfer_trtllm_fp4_routed_moe`:调用 FlashInfer MoE 内核时改回从标准字段取数据(`layer.w13_weight.data`、`layer.w2_weight.data` 等)。

关键 diff 摘录:
```diff
-            layer.gemm1_weights_fp4_shuffled = Parameter(
+            layer.w13_weight = Parameter(
                 gemm1_weights_fp4_shuffled, requires_grad=False
             )
...
-            # Clean up weights that won't be used by TRT-LLM
-            del layer.w2_weight
...
-        gemm1_weights=layer.gemm1_weights_fp4_shuffled.data,
+        gemm1_weights=layer.w13_weight.data,
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响改名合入后使用 TRTLLM FP4 MoE(nvidia 模型,如 DeepSeek-R1-0528-FP4-v2)且需要权重 offloading 的部署。
- 回归面:三个量化相关文件的 TRT-LLM FP4 MoE 分支。恢复后:权重数据本身(shuffled fp4)不变,仅存放字段名恢复标准名;offloading 重新可用;不再 `del` 原 w13/w2 权重意味着显存占用与改名前(含原权重的)状态一致。作者以 gsm8k 准确率验证与之前持平(0.9484/0.9439)。

## 关联
[#29941](https://github.com/vllm-project/vllm/pull/29941)(正文引用的被破坏的 weight offloading 功能 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
