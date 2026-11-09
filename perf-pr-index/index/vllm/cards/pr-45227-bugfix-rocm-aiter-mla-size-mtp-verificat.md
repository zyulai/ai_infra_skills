# PR #45227: [Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dtype

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chaeminlim-mb |
| 合并 | 2026-07-31 |
| 分类 | Speculative Decoding（tier: major） |
| Label | bug, documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, mistral, rust, kimi |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/45227 |

## 一句话结论

修正 ROCm AITER MLA 按真实 MTP qlen/dtype 定尺寸，GSM8K(DeepSeek-R1,TP8,MTP3) 由 0.835 升至 0.970。

## 动机

在 ROCm AITER MLA 上，MTP verification decode 的真实 qlen 为 `num_speculative_tokens + 1`（MTP3 时为 4），但此前 metadata 被当作普通 qlen=1 decode 来定尺寸：`get_mla_metadata_info_v1` 收到 `qo_len=1`，kernel 内部再对 qlen>1 重算 metadata。正文指出结果是 verification logits 出错、accept/reject 步骤偏离目标分布、accuracy 下降，并明确「传入 q/kv dtypes（#46997）本身并不能修复」。这是一个正确性 bugfix，而非吞吐优化。

## 优化点

修复思路是用真实 MTP qlen 与实际 Q/KV dtype 来定 decode metadata 尺寸：KV cache 为 FP8 时 Q 也按 FP8 处理。正文给出策略——qlen ≤ 4 保留原生 dense layout 与持久的 per-request causal boundaries；qlen > 4 回退到 qlen=1 的 split rows；CUDA-graph 的 pad rows 由合成的 `seq_lens`/`qo_indptr` 补齐。改动范围限定在 `rocm_aiter_mla.py`（ROCm AITER-only）及其单测，draft 侧 MTP（local-argmax `get_top_tokens`、`spec_step_idx` threading、attention head-count grouping）属另一事项、不在本 PR。diff 中 `AiterMLAMetadataBuilder.__init__` 依据 speculative_config 计算 `_mtp_decode_qlen = num_speculative_tokens + 1`（否则 1），并替代原常量 `1` 传给 `get_mla_metadata_info_v1`；q_dtype/kv_dtype 改为按 KV cache dtype 映射（FP8→fp8，否则按 kv_cache_spec.dtype 取 fp16/bf16），并持久化 `_mla_q_dtype`/`_mla_kv_dtype` 供 decode build 使用（避免 gfx950 fp8 nhead=32 fold 路径的 split/reduce metadata 出错）；另新增 `_uniform_padded_mtp_qo_len` 为 full-CG 的 dummy 请求合成 uniform qlen 的 pad 行。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla.py: 按真实 MTP qlen 与 Q/KV dtype 定 decode metadata 尺寸，新增 _uniform_padded_mtp_qo_len 处理 CUDA-graph pad
tests/v1/attention/test_rocm_aiter_mla_mtp_split.py: 新增单测，覆盖 qlen 策略、native/split 路由、FP8 metadata 定尺寸、CG pad 合成与 page-index 展开

# 机制摘录 (rocm_aiter_mla.py: get_mla_metadata_info_v1 调用)
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            self._mtp_decode_qlen,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
```

## 性能声明

PR 正文未给出吞吐/延迟类定量性能数据，给出的是 accuracy 结果。Test Plan 条件为 TP8、DeepSeek-R1 FP8、MTP3、greedy 下的 GSM8K（openai/gsm8k test、exact-match、n=200）：在 upstream `main` 上同一 build、仅 Python 不同，未修复为 0.835、修复后为 0.970；在 MI300X、TP8、MTP3 的本分支重跑得到 0.98 strict-match。正文还说明未修复时第 3 个 MTP 位置 acceptance 塌到约 0 且输出退化，修复后各位置 acceptance 保持约 0.80 / 0.50 / 0.30；多卡 serving accuracy 为手工运行、未进 CI。

## 验证与回归风险

Test Plan 命令为 `ruff check vllm/v1/attention/backends/mla/rocm_aiter_mla.py tests/v1/attention/test_rocm_aiter_mla_mtp_split.py` 与 `python -m pytest tests/v1/attention/test_rocm_aiter_mla_mtp_split.py -q`（ROCm-gated、CPU 上验证策略/算术）。单测 11 passed（CPU，kernel 被 mock），覆盖 qlen policy、native vs split 路由、FP8 metadata 定尺寸、CUDA-graph pad 合成，以及 `_expand_mtp_decode_page_indices_kernel` 输出与手工推导的 per-row causal indices 对照。改动仅落 ROCm AITER MLA 后端及其单测，正文明确 draft 侧 MTP 与其它后端未改；新增的 qlen>4 split 回退与 CG pad 合成是新路径。验证硬件为 MI300X，正文未注明其它 ROCm 硬件、其它精度或后端的回归结论。

## 关联

PR 正文提及前序 #46997（传入 q/kv dtypes），并指出单靠它不足以修复；draft 侧 MTP 相关改动被说明为「separate concern、不在本 PR」，其 PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
