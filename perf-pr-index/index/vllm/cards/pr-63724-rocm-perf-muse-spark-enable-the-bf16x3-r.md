# PR #63724: [ROCm][Perf][Muse Spark] Enable the bf16x3 router GEMM on gfx942

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-11-11 |
| 分类 | Attention / Kernel(tier: major) |
| Label | rocm, ready |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63724 |

## 一句话结论

为 Muse Spark 的 bf16x3 router GEMM 增加在 MI300X 上调优的 CDNA3 tile 表并打开 gfx942 门控:8,192 tokens 每层 448 → 289 µs(1.55x),约 10 ms / 8k prefill step(62 层),serving total throughput +0.5–1.2%。

## 动机

Muse Spark AMD 模型用 `router_bf16x3_gemm` 计算 FP32 router:三个 BF16 平面精确求和等于 FP32 weight,乘在 BF16 matrix cores 上。它此前被门控在 CDNA4(`_supports_router_bf16x3_device`)因为 tile 表只在彼处调优;MI300X(gfx942)上 router 因此 fallback 到 FP32 hipBLASLt GEMM(在 FP32 拷贝的 activation 上),在 TP8 的 691 ms 60k-prefill step 中占约 27 ms(router 加 cast)。kernel 本身是纯 Triton,但既有两个 tile(`≤1024`-token bucket,`BLOCK_M=128, BLOCK_N=64, BLOCK_K=128`)需要 80 KiB LDS,而 MI300X 只有 64 KiB,launch 失败。

## 优化点

- 新增 `_CONFIGS_CDNA3`(在 MI300X 上以 `triton.testing.do_bench` 对 `BLOCK_M ∈ {32,64,128} x BLOCK_N ∈ {32,64,128} x BLOCK_K ∈ {64,128} x splits ∈ {4,8,16,32} x num_warps ∈ {4,8}` 逐 token 数调优;跳过 `K % (splits * BLOCK_K) != 0`、`BLOCK_M` 大于 batch 所需、以及 launch 失败——即 80 KiB-LDS tile 在 MI300X 上被自然淘汰——的 tile;每 bucket 取最快 tile 并对照 FP64 检查 max error 与 top-8 agreement)。
- `_device_configs()`(functools.cache)按 `get_cdna_version()` 选表:`_CONFIGS_CDNA3`(CDNA3)或 `_CONFIGS_CDNA4`(原表改名,CDNA4),无默认——没有 tuned 表的架构直接 raise 而不是借用别的表;`supports_router_bf16x3` 检查运行设备自己的表。
- 门控打开为 `get_cdna_version() in (3, 4)`;两处均 lazy import `vllm.platforms.rocm`,op 模块在其他平台保持可导入;`_ROUTER_BF16X3_MIN_TOKENS = 16` 保留(MI300X 上 32 tokens 处与 FP32 持平,其余测量点领先)。

## 关键改动

```text
vllm/models/muse_spark/amd/ops/router_bf16x3.py: _CONFIGS 拆为 _CONFIGS_CDNA4/_CONFIGS_CDNA3 + _device_configs() 按 CDNA 版本选表
vllm/models/muse_spark/amd/model.py: _supports_router_bf16x3_device 由 on_gfx950() 改为 get_cdna_version() in (3, 4)
```

```python
# Tuned on MI300X (CDNA3, gfx942). Its 64 KiB of LDS rejects some of the CDNA4 tiles.
_CONFIGS_CDNA3 = (
    (16, 32, 32, 64, 32, 4),
    (128, 64, 64, 128, 32, 4),
    (256, 64, 64, 128, 16, 4),
    (512, 64, 64, 128, 8, 4),
    (1024, 128, 128, 64, 16, 8),
    (2048, 128, 128, 64, 8, 8),
    (None, 128, 128, 64, 4, 8),
)


_CONFIGS = {3: _CONFIGS_CDNA3, 4: _CONFIGS_CDNA4}


@functools.cache
def _device_configs() -> tuple[tuple, ...]:
    from vllm.platforms.rocm import get_cdna_version

    # No default: an architecture without a tuned table must not borrow one.
    return _CONFIGS[get_cdna_version()]
```

## 性能声明

- Kernel(MI300X 单卡,router shape N = 256、K = 8,192,BF16 activation;"FP32" 为 `F.linear(x.float(), w)` 即现 gfx942 路径含 cast):16 tokens 19.3 → 17.1 µs(1.13x);64 20.9 → 18.1(1.15x);128 26.9 → 20.0(1.35x);256 33.6 → 25.2(1.33x);512 48.4 → 36.2(1.34x);1,024 73.0 → 52.9(1.38x);2,048 121.7 → 83.5(1.46x);4,096 230.2 → 140.6(1.64x);8,192 448.5 → 289.2(1.55x)。16–8,192 tokens 区间整体 1.1–1.6x。
- 8,192 tokens 每层约省 159 µs,62 层约合 10 ms / 8k prefill step。
- 精度:比 FP32 路径更接近 FP64 参考(8,192 tokens max abs error 7.4e-6 vs 3.3e-5),所有测试行 top-8 expert 选择一致。
- Serving(amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8,TP8,8x MI300X,`--moe-backend triton`,`vllm bench serve` random dataset、1k output tokens,三对 base/PR,表为均值与 per-pair range):8k c16 +0.5%(4,495 → 4,518 tok/s)、8k c32 +0.6%、8k c64 +0.8%、16k c16 +1.1%、16k c32 +0.0%(−0.1 到 +0.1%)、16k c64 +1.2%(8,703 → 8,807);16–64 并发下 +0.5–1.2%(均值,三 run 均为正),除 16k c32 持平;1k 输入在 ±1% 噪声内;1–8 并发在 −0.5 到 +0.6%。median TTFT 每个 8k/16k 点约低 1–2%。
- GSM8K 5-shot(lm-eval,flexible-extract)0.9545 ± 0.0057 → 0.9522 ± 0.0059(一个标准误内不变)。

## 验证与回归风险

- 既有单测 `tests/kernels/test_muse_spark_router_bf16x3.py`(1、33、300、1,100、5,000 tokens,对 FP64 相对误差 < 2e-6)在 MI300X 通过;此前 300 与 1,100 token case 会选中超 LDS 的 tile(`OutOfResources: Required: 81920, Hardware limit: 65536`)。
- 与 Muse Spark loading、logits、shared-RoPE 模型测试共 20 passed(`vllm/vllm-openai-rocm:nightly-5d35ce3934`,挂载本分支文件)。
- 正文附完整 tuning 脚本(`tune_router_bf16x3.py`),在目标架构单卡上运行即可重建每 bucket 的 `best` tile。
- 风险:CDNA2(gfx90a)仍走 FP32 路径;无 tuned 表的架构会 raise(设计上不允许借表);kernel 数值与 FP32 路径不同(bf16x3 拆分),router 语义由 FP64 对照与 top-8 agreement 保证。

## 关联

无(正文未引用其他 PR 号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
