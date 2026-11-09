# PR #64852: [Perf][MiniMax-M3] (NVIDIA-only) Fused norm+quant MXFP8

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-11-02 |
| 分类 | Speculative Decoding \| tier: major |
| Label | nvidia, quantization, minimax |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/64852 |

## 一句话结论

新增 Triton 融合 norm+MXFP8 quant，GB300 上 M=1–16K 快 1.39–1.60×（M=16K 达 6.46 TB/s）。

## 动机

为简化集成，新增一个 Triton 融合 norm+quant MXFP8 kernel，除 MXFP8 外还覆盖不量化的 callsites：QKV proj 前的 norm+quant、Input norm、final norm、MTP enorm/hnorm；dspark drafter 的 per-head norms 仍用 FlashInfer kernel（非 2-D 输入）。原先 BF16 norm 后再单独 MXFP8 quantize 会重复读取整份 normed activation。

## 优化点

`MiniMAXGemmaRMSNorm.forward` 对 2-D 输入改调 `ops/norm_mxfp8.py` 的 `gemma_rmsnorm_mxfp8(x, weight, eps, residual=None, mxfp8=True)`；非 2-D（如 dspark drafter 的 (16384,4,128)）保留 FlashInfer。`forward_input_norm` 在 `get_input_quant_key(qkv_proj) == kMxfp8Dynamic` 且有支持的 FlashInfer MXFP8 linear kernel 时传 `mxfp8=True`。kernel 用 `triton.jit(do_not_specialize=["M"])`，hidden size 为 `tl.constexpr`、token 数不特化（共 2 个编译变体：MXFP8 on/off），以 `res_ptr`/`sf_ptr` 是否为 None 特化而非 constexpr flags，要求 2-D row-contiguous 输入（支持任意 row stride）；带 `sf` 时输出 E4M3 数据 + 128×4 swizzled 布局的 UE8M0 scales，取代 BF16 norm 后单独的 MXFP8 quantize；PDL：每个 CTA 在全局访问前 `gdc_wait()`、随后 `gdc_launch_dependents()` 让 consumer GEMM 重叠。TP>1 保留 fused all-reduce + norm 路径，其 TP1 分支也调 norm module 走 Triton。仅限 MiniMax-M3 NVIDIA，无新 flag 或依赖。实现细节上 `MXFP8_BLOCK=32`、`SF_ROW_TILE=128`，scale 按 `[m_tile, k_tile, 32, 4, 4]` 的 swizzle 布局写出、padding 行写 0；`_consumes_mxfp8_128x4` 仅当 consumer linear 的 kernel 为 `FlashInferCutedslMxfp8LinearKernel` 或 `FlashInferCutlassMxfp8LinearKernel` 时返回 True。

## 关键改动

```text
vllm/models/minimax_m3/nvidia/ops/norm_mxfp8.py: 新增 Triton _gemma_rmsnorm_mxfp8_kernel 与 gemma_rmsnorm_mxfp8（164 行新文件）。
vllm/models/minimax_m3/nvidia/model.py: MiniMAXGemmaRMSNorm.forward 对 2-D 输入走 Triton kernel；新增 _consumes_mxfp8_128x4 判定 consumer linear。
tests/models/minimax_m3/test_norm_mxfp8.py: 与 FlashInfer norm / 参考 MXFP8 quantize 的逐位与单 ulp 对比测试。
```

```python
    cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < H
    x = tl.load(x_ptr + row * x_stride0 + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    if res_ptr is not None:
        r = tl.load(res_ptr + row * res_stride0 + cols, mask=mask, other=0.0)
    h = x.to(tl.float32)
    if res_ptr is not None:
        h += r.to(tl.float32)
        tl.store(res_ptr + row * res_stride0 + cols, h, mask=mask)
    var = tl.sum(h * h) / H
    rstd = tl.rsqrt(var + eps)
```

## 性能声明

微基准：GB300 SM103、H=6144 BF16、CUPTI cold-L2 CUDA-graph 中位数、3 rotated rounds，BW 按 7.03 bytes/element，实测 streaming ceiling 6.76 TB/s。Fused MXFP8 norm vs FlashInfer norm + 独立 MXFP8 quantize 链：M=1 4.64→3.33 µs（1.39×）、4 5.15→3.26（1.58×）、16 5.28→3.30（1.60×）、64 5.41→3.68（1.47×）、256 7.55→5.25（1.44×）、1024 16.10→11.39（1.41×）、4096 46.27→31.78（1.46×）、16384 169.26→109.54 µs（1.55×，6.46 TB/s，约 ceiling 的 96%）。BF16-out norm 对 FlashInfer：除 plain M=1024 为 0.99×（噪声级，独立复测顺序翻转）外均更快。E2E（`vllm bench serve`，MiniMax-M3.1-Flash on GB300，base=`upstream/main`，2 full runs/arm，decode 用 `deep_gemm_mega_moe`、两配置均用 dspark 投机解码）：Prefill PP2 TP1、100k 随机前缀 + 8k 输入、1 输出 token，TTFT p50 c=1 288.0→286.5–286.9 ms（0.4–0.5%）、c=16 1835.9–1836.0→1820.4–1820.6 ms（0.8–0.9%），c=4 中性；Decode DEP4、60k 缓存前缀、1024 输出 token，TPOT p50 c=32 中性、c=128 11.75–12.03→10.72–10.88 ms（8.8–9.6%）、c=512 22.06–22.95→20.87–21.23 ms（5.4–7.5%）。GSM8K（MiniMax-M3.1-Flash，PP2 TP1 on 2× GB300，前 500 例、5-shot、temp 0、max_tokens 8192）：0.9380→0.9360（1 题差异，run-to-run 噪声内）。

## 验证与回归风险

`.venv/bin/python -m pytest tests/models/minimax_m3/test_norm_mxfp8.py -v` → 13 passed on GB300；`pre-commit` 全 hooks 通过。精度声明：residual / UE8M0 scales 逐位一致、E4M3 code 距离 ≤1、BF16-out 在 1 BF16 ulp 内。可能影响路径：MiniMax-M3 NVIDIA 的 2-D Gemma RMSNorm（含 QKV 前 norm、input/final norm、MTP norms、TP1 分支的 fused all-reduce helper）；非 2-D 的 dspark drafter per-head norms 仍走 FlashInfer，不在改动范围；作者称 model-local、无新 flags 或依赖。

## 关联

正文 Duplicate-work check 列出相关但不同的 PR：#62509（ROCm M3.1）、#61073（通用 QuantizedActivation 融合契约，仍在迭代）、#64723（TP>1 Qwen3.5 融合，互补）、#61783（compile-pass NVFP4 融合，格式/机制不同）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
