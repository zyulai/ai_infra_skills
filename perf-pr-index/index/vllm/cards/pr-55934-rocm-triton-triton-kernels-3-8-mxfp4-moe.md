# PR #55934: [ROCm] triton+triton_kernels 3.8 mxfp4 MoE support (gpt-oss + DeepSeek-V4)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Rohan138 |
| 合并 | 2026-09-17 |
| 分类 | MoE \| tier: major |
| Label | rocm, ready, ci/build, deepseek, gpt-oss, kv-connector, quantization, DSv4 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/55934 |

## 一句话结论

适配 triton_kernels 3.8 mxfp4 MoE；MI300X TP8 gpt-oss-120b triton c128 达 5443 tok/s。

## 动机

PR 正文说明，`triton_kernels` 3.8 重组了 mxfp4 MoE API：`matmul_ogs`→`matmul`、`RoutingData`→`RaggedTensorMetadata`、`weight_scale`→`b_mx_scale`，并移除了 in-kernel scatter+topk-reduce，既有 gpt-oss 与 DeepSeek-V4 的 MoE 路径需要适配。同时要 forward-fix 受此影响的 `AITER_MXFP4_W4A{8/16}` backend preparation（为 PR #50605 预置），并把 `Dockerfile.rock` 的 triton_kernels commit 升到 ROCm `releases/internal/3.8.x`，因为镜像 `vllm/vllm-openai-rocm:nightly-rocm100` 已带 triton 3.8。

## 优化点

为 gpt-oss 与 DeepSeek-V4 增加 3.8 triton_kernels 的 MoE forward，通过共享的 `get_triton_kernels_version` dispatcher 做版本门控（版本为 `"3.8"` 时从 `triton_kernels.matmul` 导入 `PrecisionConfig`，否则仍走 `matmul_ogs`）；forward-fix 其他受影响的 `AITER_MXFP4_W4A{8/16}` backend preparation；`Dockerfile.rock` 把 `ROCM_TRITON_KERNELS_COMMIT` 由 `0f380657dbf3ee86eb57558ff71df24f03b5d4e7` 改为 `669b31acc1dd1b3fd93286afd5db67f65d9f7557`，并把 cmake TAG 不一致时的硬检查放宽为 WARNING。可见 diff 还涉及 experts 的 aiter/gpt_oss triton 实现、oracle/mxfp4、mxfp4_utils 与 import_utils 的适配。

## 关键改动

```text
docker/Dockerfile.rock: ROCM_TRITON_KERNELS_COMMIT 升级到 669b31ac，cmake TAG 不一致改为 WARNING 而非失败
vllm/model_executor/layers/fused_moe/config.py、vllm/utils/import_utils.py: 用 get_triton_kernels_version() 门控，3.8 走 triton_kernels.matmul 的 PrecisionConfig，旧版仍走 matmul_ogs
vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py、aiter_mxfp4_w4a16_moe.py、aiter_mxfp4_w4a8_moe.py: 适配 3.8 mxfp4 MoE API 与 backend preparation
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py、vllm/model_executor/layers/quantization/utils/mxfp4_utils.py: oracle 与量化工具随版本调整
tests/kernels/moe/test_gpt_oss_triton_kernels.py: 版本判断由 use_legacy_triton_kernels 改为 triton_kernels_version == "3.5.1"
```

```python
_triton_kernels_version = get_triton_kernels_version()
if _triton_kernels_version is not None:
    try:
        if _triton_kernels_version == "3.8":
            from triton_kernels.matmul import PrecisionConfig
        else:
            from triton_kernels.matmul_ogs import PrecisionConfig
```

（摘自 `vllm/model_executor/layers/fused_moe/config.py`：按 triton_kernels 版本选择导入路径。）

## 性能声明

正文 Benchmarks 条件：TP=8，镜像 `vllm/vllm-openai-rocm:nightly-rocm100`（triton 3.8）+ triton_kernels `669b31ac`；Throughput = `vllm bench serve` Output token throughput (tok/s)，random ISL/OSL 1024/1024、`--ignore-eos`、temperature 0；Accuracy = gsm8k 5-shot、limit 200、flexible-extract、`--apply_chat_template`。gfx942 (MI300X) TP：gpt-oss-120b `aiter_triton_mxfp4_bf16` (W4A16) gsm8k 0.925，c1/c8/c32/c64/c128 = 264/1416/3946/6248/9319；`triton`（gfx942 autotune）gsm8k 0.920，c1/c128 = 106/5443；DSV4-Flash `aiter_triton_mxfp4_bf16` c1/c128 = 73/3529。gfx950 (MI355X) TP：gpt-oss-120b `AITER_MXFP4_FP8` (W4A8) gsm8k 0.890，c1/c8/c32/c64/c128 = 362/2168/7281/12366/19114；`triton` (gfx950 pin) c1/c128 = 205/9114。正文结论：gfx950 上 `aiter` 在 conc1…128 均优于 triton_kernels 3.8（gpt-oss 约 1.4–2.0×，DSV4 约 1.2–1.5×），CK `aiter` 路径对 DSV4 领先。

## 验证与回归风险

正文 Benchmarks 同时给出 gsm8k 精度与 Boot 状态，属于验证与限制信息：DSV4-Flash × `triton` (fused) 因 SWIGLUOAI-only fused path 拒绝 SILU 而 Boot ❌，需改用 `triton_unfused`；EP 模式下 `aiter_triton_mxfp4_bf16` 报 GPU memory access fault（`_moe_gemm_a16w4` 在 engine init）、`AITER_MXFP4_FP8` 报 quark loader IndexError（无 EP index remap）。diff 中另有 `test_gpt_oss_triton_kernels.py` 的版本判断改动。PR 未注明回归风险与其他模型/精度路径。

## 关联

正文提及 https://github.com/vllm-project/vllm/pull/50605（PR #50605），说明本 PR 的 forward-fix 是为该 PR 预置；未给出其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
