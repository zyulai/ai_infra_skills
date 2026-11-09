# PR #42017: [AMD][V4.1][*/N] OPUS sparse prefill on gfx950 through layout conversion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-10-01 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | deepseek, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/42017 |

## 一句话结论

gfx950 上 DSV4.1 sparse prefill 换用 aiter OPUS kernel,kernel 级快 1.4-1.7x,TTFT 最高 -29.6%。

## 动机

在 gfx950 上,DeepSeek-V4.1 的 prefill attention 一直跑 aiter_sparse 的 decode kernel——这是当时唯一能直接读 V4.1 radix KV cache 原生布局的 aiter kernel。aiter 的 OPUS sparse prefill kernel(`pa_sparse_prefill_opus`)在 16384-token chunk 上快 1.4-1.7x,但它期望完全不同的布局:连续 bf16 行(512 维/行,每个来源一个 buffer)+ CSR 索引(每项都是有效行);而 V4.1 radix cache 是 fp8 nope + scales + bf16 rope 打包成 584 B/token、256-token 页,索引列表为固定宽度、-1 padded,SWA 与 compressed history 还分属不同池。布局差异是此前用不了 prefill kernel 的根本原因。

## 优化点

核心机制是"在线布局转换 + 一次转换多方复用":把 chunk 实际读到的 KV 行用 `dequantize_k_cache_paged` 反量化到 bf16 workspace,再用 Triton kernel `combined_to_csr` 把 -1 padded 的固定宽度索引列表转成 CSR(丢弃 -1 洞和越界项,rebase 到窗口起点)。V4.1 的 ratio-1/2 层共享其 kv_source 层的 compressed KV 与 index_source 层的 top-k,所以 dequant 与 CSR 转换按源层只做一次、后续层复用(`_OpusPrefillState` 以 `data_ptr` 判断缓存命中),转换开销合计低于 attention 时间的 5%。设置三重准入门槛:`q.shape[0] >= 1024` token(dequant 才摊得开)、每 query 行的 history 行数上限(ratio 1: 256、ratio 2: 128)、workspace 不超 1 GiB;draft、context parallel、SWA bounded replay、超长压缩历史等场景回退 decode kernel。另外把 raw top-k 缓冲改为 -1 填充(原为 `empty_like`),杜绝 OPUS 读到 stale index。默认关闭,`SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1` 开启。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/opus_sparse_prefill_hip.py(新):
  combined_to_csr(Triton)-1 padded 列表转 CSR;opus_sparse_prefill 封装
  pa_sparse_prefill_opus 双源(SWA + compressed history)attention
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py:
  _opus_prefill_applies 准入判断、_forward_prefill_opus 主路径、
  _opus_compressed_inputs 按源层缓存 dequant/CSR、_drop_opus_topk_csr 失效处理
python/sglang/srt/environ.py: 新增 SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL(默认 False)
test/registered/attention/unittests/dsv4/test_dsv41_opus_sparse_prefill_hip.py(新)
```

```python
# OPUS prefill applies from this many query rows; below it the decode kernel
# splits KV and the dequant is not amortized.
_OPUS_PREFILL_MIN_TOKENS = 1024
_OPUS_PREFILL_MAX_ROWS_PER_TOKEN = {1: 256, 2: 128}
# One ratio's bf16 workspace stays within 1 GiB.
_OPUS_PREFILL_MAX_WORKSPACE_ROWS = 1 << 20
```

## 性能声明

Kernel 级(gfx950,chunk 16384):32 heads/rank 下 baseline 125.3-139.0 ms vs OPUS 73.7-96.4 ms(prefix 0-128K),比值 0.59x-0.69x;16 heads 下 0.61x-0.72x,即快约 1.4-1.7x。E2E(DeepSeek-V4.1-Flash, TP2, MI355X, DSpark, random 8000 in / 1000 out,中位数):conc=1 TTFT -8.0%、output throughput +0.8%;conc=4 TTFT -10.9%、throughput +2.6%;conc=16 TTFT -29.6%、TPOT -4.7%、throughput +5.7%;conc=64 TTFT -6.3%、TPOT -4.8%、throughput +6.1%。精度:gsm8k 1319 = 0.908。

## 验证与回归风险

新增 `test_dsv41_opus_sparse_prefill_hip.py`(register_amd_ci,MI355X 1-GPU,stage-b),覆盖:CSR 转换对照逐行 Python 参考、双源/单源 OPUS attention 对照 fp32 参考(TOL 1e-2)、raw top-k buffer 的 -1 填充不变量。风险面:路径默认关闭,关闭时无影响;开启时 KV 经 fp8→bf16 dequant,精度受 bf16 限制;`_drop_opus_topk_csr` 与 indexer 重写 top-k 的交互需保持一致;回退条件(页大小、CP、SWA replay 等)若遗漏新场景可能跑错路径。

## 关联

同区域:environ 中相邻的 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`(DSV4 sparse prefill 的另一条 HIP 路径);pack 未列出其他相关 PR 编号,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
