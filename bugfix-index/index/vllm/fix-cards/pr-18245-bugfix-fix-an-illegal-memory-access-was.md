# PR #18245: [Bugfix] fix `an illegal memory access was encountered` of marlin kernel + act_order

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18245](https://github.com/vllm-project/vllm/pull/18245) |
| 作者 | jinzhen-lin |
| 合并日期 | 2025-05-16 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论

Marlin kernel 在 act_order 下越界读 `g_idx` 触发 CUDA illegal memory access,加 `slice_k_start < prob_k` 边界检查修复。

## 问题现象

正文未描述具体现象(正文仅链接了 PR #18147)。从标题看,症状是在启用 act_order 的 GPTQ/Marlin 量化模型上,kernel 运行报 CUDA 错误 `an illegal memory access was encountered`(非法内存访问)。相关测试模型 `TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ` 的 `gptq-8bit-32g-actorder_True` 配置此前在 `tests/weight_loading/models.txt` 中被注释禁用,本 PR 将其重新启用。

## 根因

正文未披露根因。(由 diff 推断)在 Marlin kernel 的 `has_act_order` 分支中,`slice_k_start += tb_k * stages` 推进后没有检查是否已越过 `prob_k`(K 维总长度),就执行 `g_idx[slice_k_start]` 和 `g_idx[last_g_idx]` 读取;当 `slice_k_start >= prob_k`(例如最后一个 slice 的迭代数耗尽时)会造成对 `g_idx` 的越界读取,触发非法内存访问。此外旧代码在越界情况下仍会推进 `slice_k_start_shared_fetch` 并可能调用 `__syncthreads()`,部分线程提前退出时会造成不一致。

## 修复方案

在两处 Marlin 模板文件(`csrc/moe/marlin_moe_wna16/marlin_template.h` 与 `csrc/quantization/gptq_marlin/marlin_template.h`)的 `has_act_order` 分支中,把 act_order scale 的预取逻辑包进边界检查 `if (slice_k_start < prob_k)`,越界时不再读取 `g_idx`、不再推进 `slice_k_start_shared_fetch`,也不再多做 `__syncthreads()`:

```diff
       if constexpr (has_act_order) {
         slice_k_start += tb_k * stages;
-        slice_k_start_shared_fetch += tb_k * stages;
-        int first_group_id = g_idx[slice_k_start];
-        ...
-        if (last_group_id >= sh_first_group_id + sh_num_groups) {
-          fetch_act_order_scales_to_shared(false, first_group_id, last_group_id);
-          __syncthreads();
+
+        if (slice_k_start < prob_k) {
+          slice_k_start_shared_fetch += tb_k * stages;
+          int first_group_id = g_idx[slice_k_start];
+          int last_g_idx = slice_k_start + stages * tb_k * 2;
+          if (last_g_idx >= prob_k) {
+            last_g_idx = prob_k - 1;
+          }
+          int last_group_id = g_idx[last_g_idx];
+          if (last_group_id >= sh_first_group_id + sh_num_groups) {
+            fetch_act_order_scales_to_shared(false, first_group_id,
+                                             last_group_id);
+            __syncthreads();
+          }
         }
       }
```

同时把 `tests/weight_loading/models.txt` 中被注释的 `gptq-8bit-32g-actorder_True` 测试项重新启用作为回归验证。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:两个 Marlin kernel 模板(MoE wna16 与 GPTQ Marlin)在 `has_act_order` 路径上的执行逻辑;所有使用 act_order 的 Marlin/GPTQ-Marlin 量化推理都会经过该分支。改动在正常范围内行为不变(条件恒真时执行相同代码),仅在临界 slice 上跳过预取。

## 关联

- https://github.com/vllm-project/vllm/pull/18147

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
