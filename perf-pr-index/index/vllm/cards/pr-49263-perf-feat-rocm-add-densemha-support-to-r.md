# PR #49263: [Perf] [Feat] [ROCm] Add densemha support to ROCm AITER Sparse MLA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tjtanaa |
| 合并 | 2026-10-12 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, ci/build, v1, deepseek, mrv1-only, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 4 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/49263 |

## 一句话结论

ROCm AITER Sparse MLA 短 prefill 改 dense MHA，DeepSeek-V3.2 C8 TTFT 526.61→480.76 ms（-8.69%）。

## 动机

ROCm AITER Sparse MLA 后端此前对短 prefill 也走稀疏 MQA。PR 参照 #47327，为短 prefill 增加 dense MHA：prefill 长度不超过 sparse top-k 阈值（DeepSeek-V3.2 为 2048 tokens）时用 dense attention，更长 prefill 与 decode 仍用稀疏 MQA；并接入融合的 DeepSeek-V3.2 路径，使混合 prefill/decode 批次拿到正确 metadata。

## 优化点

在 ROCm AITER Sparse MLA 后端按 prefill 长度分流：≤2048（DeepSeek-V3.2 的 sparse top-k 阈值）走 dense MHA，>2048 及 decode 走稀疏 MQA。集成融合 DeepSeek-V3.2 路径以处理混合 prefill/decode 批次 metadata。测试计划要求对比 BF16/FP8-KV 引擎输出与 logprobs 对齐强制稀疏 MQA，覆盖不等长/分块 prompt、2048 边界、混合 prefill/decode，并把真实 pretrained dense/sparse 路径的 trace 与计时分开，在 concurrency 1/8/32 下对比 short/long-prompt serving 与 matched GSM8K/更难任务精度；另运行 MI300/MI355 CI 收集的 metadata、sparse-attention、sink 测试。

## 关键改动

```text
.buildkite/test_areas/attention.yaml: 把 ROCm AITER sparse MLA 相关测试纳入 CI
tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py: metadata 同步测试
tests/v1/attention/test_rocm_glm5next_sparse.py、test_sparse_mla_backends.py: 稀疏 MLA 后端行为测试
（核心实现文件 4 个因体积被 dropped from diff，pack 内仅保留测试与 CI 文件）
```

## 性能声明

Pretrained runs：official DeepSeek-V3.2，8× AMD gfx950/TP8，FP8 weights、BF16 activations/KV，AITER，eager，prefix caching off（ROCm 7.2.3 / Torch 2.13.0+git733fca1 / AITER 0.1.23 / Triton 3.8.0+git669b31ac）。三轮配对、固定 64-token 输出、warmup、864/864 成功计时请求。Serving short prompts：Concurrency 8 TTFT 526.61→480.76 ms（-8.69%，95% 区间 [-16.87%, -0.51%]）、TPOT/ITL 160.60→158.75 ms、Output 48.07→48.79 tok/s；Concurrency 32 TTFT 975.01→904.60 ms（-7.07%，95% [-20.12%, +5.99%]）、TPOT/ITL 143.40→139.69 ms、Output 198.15→204.27 tok/s。长 prompt TTFT 大致不变，短/concurrency-1 吞吐 -1.23%；多数吞吐区间跨零，非 compiled-mode 或 broad no-regression 结论。精度 GSM8K 5-shot strict 95.38→95.53%、30-shot 95.30→95.38%，MMLU-Pro final 80.13→79.69%、supplementary 80.36→79.24%；PR 明确声明 No equivalence or non-inferiority claim。

## 验证与回归风险

发布版本 `7abb8ef` 与已验证的 `e34b94e` 为同一 tree，baseline 为 `b538d807`。测试：187 passed, 116 skipped, 14 warnings；local staged pre-commit passed。Runtime tracing 确认 dense prefill 到 2048 tokens、2049 起 sparse、混合 dense-prefill/sparse-decode 执行。回归风险：短/concurrency-1 吞吐 -1.23%；GSM8K 对照产生 adverse candidate output（原因未明）；MMLU-Pro 为 category-balanced 诊断而非完整 benchmark，作者明确不作等价/非劣结论。

## 关联

正文提到参照前序 PR #47327；未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
