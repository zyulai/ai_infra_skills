# PR #30647: [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @elvischenv |
| 合并 | 2026-03-18 |
| 分类 | MoE（tier: major） |
| Label | ready, ci/build, gpt-oss, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/30647 |

## 一句话结论

消除 GPT-OSS MoE 前后的 pad 与 slice、恢复 AR+Norm 融合，GPT-OSS-120b TP8 con8 E2E +5%。

## 动机

正文标注依赖 Flashinfer 更新 #30993。原先 GPT-OSS 的 Flashinfer MXFP4 MXFP8 MoE 前有一个 padding op（mxfp8 quant 输出未对齐需补 pad），MoE 后有一个 slicing op（去掉 padding 的 hidden），而 slice op 还破坏了此前的 AR+Norm fusion（#28841）。改动目的：通过在 flashinfer mxfp8 quant 里设置 alignment 让量化输出张量自带 padding，并把未 pad hidden size 的输出张量直接传给 MoE kernel 以消除 slice（依赖 flashinfer PR #2217）。正文用删除线划掉了"清理 `maybe_roundup_hidden_size()` 重复逻辑"一项，即该清理未纳入最终改动。

## 优化点

在 flashinfer mxfp8 quant 设置 alignment，使输出量化张量本身就是 padded，从而省掉 MoE 前的 padding op；向 MoE kernel 传未 pad hidden size 的输出张量，省掉 MoE 后的 slicing op。由此恢复被 slice 打断的 AR+Norm 融合。Kernel timeline 对比印证：main 侧有 `triton_poi_fused_constant_pad_nd_moe_forward_0` 1.407 µs（pad）、`vllm::cross_device_reduce_1stage` 8.320 µs（ar）与 `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_slice_1` 2.336 µs（norm, slice）；PR 侧变为 `flashinfer::trtllm_allreduce_fusion` 7.744 µs（ar+norm），不再有独立的 pad 与 slice 项。

## 关键改动

```text
vllm/model_executor/layers/quantization/mxfp4.py: 在 mxfp8 quant 设置 alignment，使量化输出自带 padding
vllm/model_executor/layers/fused_moe/fused_moe_method_base.py 与 runner/default_moe_runner.py: 向 MoE kernel 传未 pad hidden size 的输出张量，去掉 slice
tests/compile/fusions_e2e/conftest.py、models.py、test_tp2_ar_rms.py: 为 gpt-oss-20b 增加 AR+RMS fusion 覆盖并置 VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
```
```text
        # TODO: remove this after finishing migration from envs to model kwargs
        if model_name == "openai/gpt-oss-20b":
            monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "1")
```

## 性能声明

GPT-OSS-120b TP8、concurrency 8、80 successful requests、81920 input / 81920 generated tokens，PR 标注 **5% E2E improvement**。Serving Benchmark（PR / main）：Benchmark duration 28.90 / 30.43 s；Request throughput 2.77 / 2.63 req/s；Output token throughput 2834.13 / 2692.11 tok/s；Total token throughput 5668.25 / 5384.22 tok/s；Mean TPOT 2.77 / 2.92 ms；Mean TTFT 50.37 / 52.61 ms；Mean ITL 54.59 / 57.47 ms。Accuracy：PR gpqa 0.7803 / aime25 0.8875，main gpqa 0.7891 / aime25 0.8875。测试条件：模型 GPT-OSS-120b、TP8、concurrency 8、80 请求、8k 输入 / 1k 输出量级，量化为 Flashinfer MXFP4 MXFP8 MoE；PR 未注明 GPU 型号与精度以外的更多细节。

## 验证与回归风险

验证包括 gpqa / aime25 精度 PR 与 main 对比、Kernel timeline 对比（证实 pad/slice 消失且 AR+Norm 融合），以及 `tests/compile/fusions_e2e` 新增 gpt-oss-20b 的 AR+RMS fusion 计数覆盖。回归风险：改动依赖 Flashinfer 更新 #30993 与 flashinfer PR #2217；影响 GPT-OSS 的 MXFP4 MXFP8 MoE 路径与 AR+Norm fusion；精度上 gpqa 略降（0.7803 vs 0.7891）、aime25 持平。正文注明"清理 `maybe_roundup_hidden_size()` 重复逻辑"一项已被划掉、不在最终改动内。

## 关联

正文明确 #30993（依赖的 Flashinfer 更新）、flashinfer#2217（依赖的 Flashinfer PR）与 #28841（此前被 slice op 破坏的 AR+Norm fusion issue）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
