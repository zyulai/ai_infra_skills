# PR #67377: [Perf][ROCm][DSv4.1] DeepSeek-V4.1-Flash: dense MXFP8 GEMMs and wo_a on AITER's tuned FlyDSL kernels, and a faster MXFP8 activation quant (MI355X)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @limou102 |
| 合并 | 2026-11-15 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, deepseek, quantization, verified |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67377 |

## 一句话结论

DeepSeek-V4.1-Flash 在 MI355X(gfx950)上把 32x32 block-scaled MXFP8 dense linears 与 grouped `wo_a` BMM 换到 AITER 按每 M 选择的 tuned FlyDSL GEMM,并把 MXFP8 activation quant 提速约 2x(输出 bit-identical);按引擎步加权,dense GEMM prefill 1.78x、decode 1.61x,无额外显存。

## 动机

DeepSeek-V4.1-Flash 以 DP8 attention + EP8 MoE(TP1 dense linears)在 8x MI355X 上服务时,其 32x32 block-scaled MXFP8 linears 跑的是 Triton block32 GEMM。该 GEMM 没有 TP1 tuning(`_default_config`),只达到 MXFP8 peak 的 21-31%。在 gfx950 上启用 AITER 后,这些 linears 与 grouped `wo_a` BMM 可改跑 AITER 的 FlyDSL GEMM(AITER 按 M 从 tuned 表选择);MXFP8 activation quant 也有约 2x 的提速空间且可保持输出 bit-identical。

## 优化点

- `RocmDotScaledMxfp8LinearKernel`:满足以下条件的 layer 走 FlyDSL——AITER linear 开启、GPU 为 gfx950、模型 bf16、weight 为 32x32 scale block 且 `K % 64 == 0`、AITER 对该 shape 有 tuned 32x32 rows(`rocm_aiter_ops.is_mxfp8_block32_bpreshuffle_gemm_tuned`)。这类 layer 在 load 时 (16, 16) in-place preshuffle,并暴露 `layer.weight_preshuffled`。
- `rocm_mxfp8_block32_gemm(_quant)`:给定 `weight_preshuffled` 时调 AITER 的 `gemm_a8w8_blockscale_bpreshuffle`;FlyDSL kernel 使用 32-bit offsets,行按 2 GiB 分块跑。未给 preshuffled 时,既有 Triton 路径完全不变。
- `rocm_dsv41_mxfp8_wo_a_bmm`:给定 `weight_preshuffled` 时调 AITER 的 `batched_gemm_a8w8_mxscale_bpreshuffle`。
- Mono decode(gfx950)按 row-major 读 dense linears:被它读取的 linears 标记 `keep_row_major_weight`,保留原权重并另存一份 preshuffled 拷贝供 AITER GEMM 使用。
- `_mxfp8_quant_triton_kernel`:每个 program 处理一个 [4, 512] tile(原为一个 [64, 32] block),算法不变、输出 bit-identical,quant 提速约 2x。
- 不变路径:其余 layer 与配置——1x32-scaled 模型、未 tune 的 shape、非 gfx950(含 gfx1250)、AITER 关闭、`VLLM_BATCH_INVARIANT`。

## 关键改动

```text
vllm/model_executor/kernels/linear/mxfp8/rocm_native.py: flydsl_enabled 门控 + load 时 in-place (16,16) preshuffle + apply_weights 传 weight_preshuffled
vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py: 新增 _aiter_gemm(gemm_a8w8_blockscale_bpreshuffle,FLYDSL_MAX_BYTES 2 GiB 行分块)
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: 新增 _aiter_wo_a_bmm(batched_gemm_a8w8_mxscale_bpreshuffle)
vllm/_aiter_ops.py: 新增 is_mxfp8_block32_bpreshuffle_gemm_tuned(按 CU 数查 AITER tuned 32x32 rows)
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py: _mxfp8_quant_triton_kernel 改为每 program [BLOCK_M, 512] tile,num_warps 4→2
vllm/models/deepseek_v41/amd/mono_decode.py: 新增 _keep_row_major_at_load
tests/kernels/quantization/test_rocm_mxfp8_linear.py: 新增 FlyDSL vs Triton 一致性测试
tests/models/test_dsv41_mono_numerics.py: mono 测试给 shared expert 标 keep_row_major_weight
```

```python
def _aiter_gemm(x, x_scale, weight_preshuffled, weight_scale, out) -> None:
    from aiter import gemm_a8w8_blockscale_bpreshuffle

    M, K = x.shape
    e8m0 = torch.float8_e8m0fnu
    rows = FLYDSL_MAX_BYTES // max(K, out.element_size() * out.shape[1])
    for m0 in range(0, M, rows):
        gemm_a8w8_blockscale_bpreshuffle(
            x[m0 : m0 + rows],
            weight_preshuffled,
            x_scale[m0 : m0 + rows].view(e8m0),
            weight_scale.view(e8m0),
            out.dtype,
            out=out[m0 : m0 + rows],
        )
```

## 性能声明

- 端到端(InferenceX aiperf agentic,CONC=256,DSpark 5 tokens):throughput_per_chip 26,714 tok/s/GPU,P90 interactivity 4.61 tok/s/user;与本 PR 上一版同节点结果(27,385 / 4.60)在 run-to-run noise 内,上一版对 main 为 +11.1% / +10.9%。单一 revision 的各 run 间差异可达 5%。
- Dense GEMM 离线(按 profiled run 每引擎步调用次数加权):prefill 189.8 → 106.7 ms per step(1.78x);decode 4544 → 2822 us per step(1.61x)。
- MXFP8 activation quant:357 → 172 us(M = 32768,K = 8192)。
- 单 GEMM(一张 MI355X,HIP-graph replay over rotating weight copies,AITER v0.1.29,linear 计时含 activation quant;main → PR,prehid M=32768 / decode 加权):wq_b (32768x1280) 2420→1413(1.71x)/ 16.16→11.84(1.36x);wo_a (G=8) 1995→956(2.09x)/ 13.13→10.47(1.25x);wo_b (5120x8192) 1980→1128(1.76x)/ 35.94→16.86(2.13x);shared expert gate_up (4608x5120) 1155→688(1.68x)/ 24.63→12.33(2.00x);shared expert down (5120x2304) 565→373(1.51x)/ 10.77→7.68(1.40x);fused wq_a|wkv (1792x5120) 611→342(1.79x)/ 9.10→8.64(1.05x);indexer wq_b (4096x1280) 280→214(1.31x)/ 6.06→5.41(1.12x);engram (25600x6144) 6921→3890(1.78x)/ 53.01→32.50(1.63x)。
- 无额外显存:weights in-place preshuffle,模型保持 52.1 GiB per GPU。
- 精度不变:GSM8K 5-shot strict-match 0.9735(main)vs 0.9727(本 PR),±0.0045;flexible-extract 0.9727 vs 0.9719。

## 验证与回归风险

- 端到端与 GSM8K 在 `43ccbef` 上跑(最终 cleanup 前的同逻辑);测试在最新代码(`564fd1c` 仅加一个 type assert)。
- 每个引用了被改模块的测试文件(142 个)main vs 本分支结果一致:6778 → 6793 passed(新增测试),双方同样 751 failed / 133 errors(需网络、缺 `ray`、CUDA-only 等)。
- `test_rocm_mxfp8_linear.py` + `test_mxfp8_kernel_selection.py` 324 passed;`test_rocm_triton_attn_dsv4.py -k wo_a` 22 passed;`test_dsv41_mono_decode.py`、`test_minimax_m3_amd_ops.py`、`test_quant_activation_contract.py` 251 passed 1 skipped;`test_dsv41_mono_layers.py -k test_mono_decode_matches_the_model`(TP2/4/8,AITER on)4 passed 1 skipped(gfx942-only case);`pre-commit run` 通过。
- 新测试 `test_rocm_mxfp8_linear_flydsl_matches_triton`、`test_rocm_mxfp8_wo_a_flydsl_matches_triton` 断言 AITER 路径与 Triton 路径一致(rel. err < 1e-3);AITER 无 tuned rows 时(按 CU 数 keyed,如 partitioned MI355X)layer 保持 Triton。
- 风险:AITER FlyDSL GEMM 仅限 gfx950 + bf16 + 32x32 scale block 且 K % 64 == 0 且有 tuned rows 的 shape;FlyDSL 32-bit offsets 引入 2 GiB 行分块;mono decode 配置对每个被读取的 linear 多保一份 preshuffled 拷贝(DP/EP 配置单份);in-place preshuffle 依赖 `keep_row_major_weight` 正确标注。

## 关联

无直接关联 PR(正文注明 AITER 无需改动:v0.1.29 已带这些 shape 的 tuned 32x32 rows)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
