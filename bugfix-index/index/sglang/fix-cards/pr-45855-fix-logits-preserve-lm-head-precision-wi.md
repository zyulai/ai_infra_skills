# PR #45855: [Fix] logits: Preserve lm_head precision with shared buffers

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45855](https://github.com/sgl-project/sglang/pull/45855) |
| 作者 | YAMY1234 |
| 合并日期 | 2026-10-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
共享 FP32 logits buffer 使 FP16/BF16 lm_head 意外输出 FP32 GEMM、舍入不一致;限制 FP32 直写仅限显式开启 FP32 的 head。

## 问题现象
正文披露:当分配了匹配的 FP32 logits buffer 时,普通 FP16/BF16 lm_head 即使未启用 FP32 lm_head,也会产生 FP32 GEMM 输出——这跳过了无 buffer 的 prefill 打分路径与 LoRA 包裹的 head 所使用的舍入(rounding),造成 decode/prefill 数值不一致。具体表现为 DSpark decode/prefill logprob mismatch 以及同期出现的 LoRA/UNO parity 失败(端到端确认尚在进行中,PR 以 draft 状态保留待 GPU 验证结果)。

## 根因
正文披露:该问题由 #45631 引入——共享 FP32 logits buffer 落地后,`_compute_lm_head` 中只要 `out`(buffer)存在就走 fp32 out_dtype 的 `torch.mm`,导致默认 dtype 的 lm_head 输出精度被 buffer 改变。(由 diff 推断)条件 `use_mm_out_dtype and (self.use_fp32_lm_head or out is not None)` 中 `out is not None` 这一分支让任何带共享 buffer 的调用都改用 fp32 GEMM,与无 buffer 路径产生一位(one ULP 级)舍入差异。

## 修复方案
1. `python/sglang/srt/layers/logits_processor.py` `_compute_lm_head`:条件由 `if use_mm_out_dtype and (self.use_fp32_lm_head or out is not None):` 收紧为 `if use_mm_out_dtype and self.use_fp32_lm_head:`,即只有显式启用 FP32 lm_head 才直接以 fp32 写入共享 buffer;默认 head 先按原始 dtype 计算,再拷入共享 buffer。显式 FP32 与 MuseSpark 既有的 FP32 override 仍保留直写。
2. `test/registered/rl/test_fp32_lm_head.py`:新增 buffered/unbuffered 精度一致(FP16/BF16)、显式 FP32 直写检查、FP32 开/关两种情况下的 CUDA graph replay 测试;低精度参考值改为普通 dtype GEMM(原先用 FP64 结果再舍入,可能差一个输出 ULP),容差不变。

```diff
-            if use_mm_out_dtype and (self.use_fp32_lm_head or out is not None):
+            # A shared FP32 buffer must not change the lm_head's output precision.
+            # Keep the same rounding as the unbuffered path unless FP32 is enabled.
+            if use_mm_out_dtype and self.use_fp32_lm_head:
                 logits = torch.mm(
```

## 影响与回归风险
- 影响窗口:#45631 引入共享 logits buffer 之后的版本(正文未给出具体版本号)。
- 回归面:所有带共享 FP32 logits buffer 的 logits 计算路径:默认 FP16/BF16 head 的输出舍入恢复为与无 buffer 路径一致;显式 `--enable-flashinfer-fp32-lm-head` 类 FP32 head 与 MuseSpark1xLogitsProcessor 路径不变;CUDA graph replay 路径中 mm 是否被调用发生变化(默认 head 不再调用 fp32 mm);测试文件参考值计算方式变更。

## 关联
- Related: #45846(https://github.com/sgl-project/sglang/issues/45846 或对应 PR)
- Introduced by: #45631(https://github.com/sgl-project/sglang/pull/45631)
- GPU 验证 run:https://github.com/sgl-project/sglang/actions/runs/39822486281 、https://github.com/sgl-project/sglang/actions/runs/39822512516

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
