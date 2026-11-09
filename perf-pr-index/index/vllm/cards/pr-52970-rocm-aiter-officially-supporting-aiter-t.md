# PR #52970: [ROCm][aiter] officially supporting aiter Triton kernels+dsv4 on RDNA3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-xavierwang |
| 合并 | 2026-10-27 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, torch.compile, deepseek, quantization, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/52970 |

## 一句话结论

把 AITER Triton FP8 路径从 RDNA4 扩到 RDNA3（gfx11），完成 dsv4 的 GPU kernel 支持。

## 动机

在 RDNA3（gfx11）上启用 AITER Triton FP8 路径，使其与 gfx12/RDNA4 对齐，并完成 dsv4 模型的完整 GPU kernel 支持。正文指出 gfx11 与 gfx12 都是 RDNA、都没有 CK build，只有 aiter 的 Triton kernels；此前 gate 只针对 RDNA4，导致 gfx11 需依赖 op 级 CPU/torch fallback（#47017 使 DSV4 能跑在 gfx11 但靠 fallback，`item()` 会导致 cudagraph capture 失败、只能用 `--enforce-eager`）。

## 优化点

`VLLM_USE_AITER=1` 时启用 AITER unified attention，优先级在 `ROCM_ATTN` 之后、`TRITON_ATTN` 之前；启用 AITER GDN kernel；把原本只针对 RDNA4 的多数 gates 放宽到全部 RDNA。核心是把两个 RDNA gate 换成统一的 `on_rdna()` 谓词（gfx11 + gfx12，减 CDNA gfx1250），使 fp8 MoE oracles、blockscale linear、quant-fusion passes、pass-manager、GDN、sparse-MLA 等下游站点对 gfx11 自动生效而无需逐处改。`_aiter_ops.py` 新增 `is_aiter_found_and_supported_on_rdna()`，RDNA gate 由 `on_rdna4` 改为 `on_rdna`，并把 `register_ops_once` 里 aiter ops 走通用路径注册（去掉 gfx11 的 `_direct_register_custom_op_()` 脆弱做法）。`rocm.py` 的 backend 优先级对 RDNA4 用 `insert(0)`、其余 RDNA 用 `append` 差异化处理。

## 关键改动

```text
vllm/_aiter_ops.py: 新增 is_aiter_found_and_supported_on_rdna()；is_rdna_aiter_enabled / is_rdna_linear_enabled / are_rdna_gdn_triton_kernels_available 由 on_rdna4 改用 on_rdna；register_ops_once 走通用注册路径。
vllm/platforms/rocm.py: _get_backend_priorities 对 RDNA 架构差异化插入 ROCM_AITER_UNIFIED_ATTN；get_default_ir_op_priority 去掉 not on_rdna4() 限制。
vllm/compilation/passes/fusion/rocm_aiter_fusion.py: GDN triton kernels 判定改用 are_rdna_gdn_triton_kernels_available()。
vllm/model_executor/kernels/linear/scaled_mm/aiter.py、layers/mamba/gdn/qwen_gdn_linear_attn.py、layers/sparse_attn_indexer.py、models/deepseek_v4/amd/rocm.py、v1/attention/ops/rocm_aiter_mla_sparse.py: 各下游 gate 放宽到 RDNA。
```

```text
def is_aiter_found_and_supported_on_rdna() -> bool:
    """RDNA (gfx11 + gfx12) analog of `is_aiter_found_and_supported()`."""
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_rdna

        return on_rdna()
    return False
```

## 性能声明

PR 正文未给出定量性能数据（无吞吐 / 加速 benchmark 数字）。Test Result 为 gsm8k 精度结果：8× AMD Radeon Pro W7900（gfx1100 / RDNA3），TP=8，Model `deepseek-ai/DeepSeek-V4-Flash-0731`，`VLLM_ROCM_USE_AITER=1`，`--enforce-eager`，`--kv-cache-dtype fp8`，dspark speculative config（5 tokens），lm_eval gsm8k 5-shot、max_gen_toks 2048：flexible-extract exact_match **0.8840 ± 0.0088**，strict-match **0.6187 ± 0.0134**；正文注明 "96/1319 empty due to the 2048-token gen budget → 0.953 on the 1223 answered"，dsv4 有 thinking module。PR 未注明吞吐 / 延迟测试条件（无 serving benchmark 数字）。

## 验证与回归风险

验证：Test Plan 给出硬件（8× W7900 gfx1100，TP=8）、模型、`VLLM_ROCM_USE_AITER=1` 的 serve 命令（含 `VLLM_USE_V2_MODEL_RUNNER=1`、`TRITON_AMD_REAL_TRUE16=1`、`PYTORCH_HIP_ALLOC_CONF`、dspark speculative config、`--enforce-eager`、max-model-len 4096）与 lm_eval 命令；Test Result 为上述 gsm8k。风险：serve 仍用 `--enforce-eager`（正文指出 `item()` fallback 会使 cudagraph capture 失败，本 PR 去掉 gfx11 的 torch-fallback 依赖）；gsm8k strict-match 偏低且 96/1319 因 2048-token gen budget 为空；`on_rdna()` 排除了 CDNA gfx1250，其行为不变；正文称 2026-08-19 复查 open PR 无重复。

## 关联

#43615（延续 / 镜像，把 RDNA gate 统一为 `on_rdna()`）、#47017（补充，去掉 gfx11 的 torch fallback 依赖）、#51555（supersedes，改用 AITER 内核而非新增 in-kernel dequant kernel）、ROCm/aiter#4764（AITER 侧 minor fix）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
