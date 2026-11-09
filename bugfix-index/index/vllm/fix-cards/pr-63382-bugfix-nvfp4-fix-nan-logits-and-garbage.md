# PR #63382: [Bugfix][NVFP4] Fix NaN logits and garbage output from uninitialized fused act+quant scales

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63382](https://github.com/vllm-project/vllm/pull/63382) |
| 作者 | nv-nikitaj |
| 合并日期 | 2026-10-29 |
| labels | bug, quantization, ready |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论

融合 `silu_and_mul_nvfp4_quant` 内核未初始化 swizzled scale 填充区,GEMM 读到脏内存产生 NaN;内核自行清零填充行列修复。

## 问题现象

NVFP4 checkpoint(如 `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4`)输出 NaN logits 与乱码(`other other other ...`);开启 logprobs 的 `/v1/chat/completions` 返回 HTTP 400 `Out of range float values are not JSON compliant: nan`。触发条件:使用融合 kernel `silu_and_mul_nvfp4_quant`(`fuse_act_quant: True`)且 scale 分配器为 `torch.empty`。作者在 8x RTX PRO 4500 (sm120)、TP=8、FlashInfer-Cutlass NVFP4 linear 后端、CUDA 13.2 源码编译环境下,用 100 次 chat+logprobs 与 100 次 completions 请求的脚本复现。

## 根因

正文披露:回归来自 #53568——它把 scale 分配器从 `torch.zeros` 换成 `torch.empty`(`torch.zeros` 此前曾在 #45739 被恢复)。融合 kernel `silu_and_mul_nvfp4_quant` 只写 swizzled block-scale buffer 的有效行,而 NVFP4 GEMM 会读填充条目,于是把陈旧内存当 scale 消费,产生 NaN logits。修复不加分配器 memset,由 kernel 自己清零填充的 scale 行列,与未融合的 `scaled_fp4_quant` 行为一致。

## 修复方案

修改 `csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu` 的 `silu_mul_cvt_fp16_to_fp4` kernel 与 host 函数 `silu_and_mul_nvfp4_quant_sm1xxa`:kernel 循环改为覆盖含 padding 的行(`sf_m = round_up(numRows, 128)`)与列(`num_padded_cols`),padding 处喂零输入;无效位置若 `sf_out` 非空则显式写 0;grid 计算改为覆盖 padding,`grid_x` 用 `computeEffectiveRows(m)`。两个使用方(`ActivationQuantFusionPass` 与 `fused_act_quant.py` 手动融合)分配器保持 `torch.empty`,无额外 memset。新增 8 个单测(文件共 20 passed)。

```diff
+      fp4_packed_t out_val{};
+      if (valid) {
+        out_val =
+            cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
+                out_silu_mul, SFScaleVal, sf_out);
+      } else if (sf_out != nullptr) {
+        // The swizzled scale tensor may be allocated with torch.empty.
+        // Explicitly initialize padded rows and columns in this kernel so
+        // downstream NVFP4 GEMMs never observe stale scale factors.
+        *sf_out = 0;
+      }
```

## 影响与回归风险

- 影响窗口:#53568(分配器改为 `torch.empty`)之后的版本,正文称本 PR 修复该回归。
- 回归面:csrc 融合量化 kernel 的行/列循环与 grid 覆盖扩大到 padding 区,输出写回仍由 `valid` 门控;影响 `ActivationQuantFusionPass` 与 `fused_act_quant.py` 两个调用方;CPU 侧无改动。

## 关联

#53568、#45739

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
