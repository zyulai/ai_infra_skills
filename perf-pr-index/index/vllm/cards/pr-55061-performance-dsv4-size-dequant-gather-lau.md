# PR #55061: [Performance][DSv4] Size dequant gather launch grid by rows

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-09-04 |
| 分类 | CUDA Graph / Compiler |
| Label | ready, deepseek, quantization, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/55061 |

## 一句话结论

DSv4 去量化 gather 内核 grid.y 按行数动态取值,BI decode 每步 gather 从 1.66ms 降到 0.51ms。

## 动机

`dequantize_and_gather_k_cache_cutedsl`(DSv4 KV cache 去量化 + gather,CuTeDSL 内核)写死了 `grid = (num_reqs, 1024)`。该内核来自 #42236,其画像场景是单请求 prefill(一次数千到 262k 行),1024 固定值彼时接近最优;但在 batch-invariant(BI,确定性)decode 中,每请求每步只有 1 个 token:SWA gather 每请求 128 行、compressed 约 1k 行,32 请求合计 4096 行摊在 32768 个 CTA 上,绝大多数空转,实测有效带宽仅 0.35–2.4 TB/s(GB200 HBM 8 TB/s),18.9us 干了 3.0us 的活。BI 稀疏 decode 每步调用该内核 84 次(paged 路径为 0 次),gather 因此进入 BI decode 关键路径。

## 优化点

把 grid.y 从编译期常量改为运行时 kernel 参数,由 wrapper 按调用形状计算:`grid.y = max(1, min(ceil(max_rows/4), ceil(8192/num_reqs)))`(max_rows = `out.shape[1] - offset`,即调用方给的每请求行数上界)。两项均有实测依据:第一项"每 warp 一行"(CTA 4 warp)是单个请求能利用的全部并行度,去掉它 32 请求 SWA decode 从 3.1us 退到 18.7us;第二项把总 CTA 数维持在约 8192 的实测吞吐平台(单请求 262144 行 gather 在 1024/4096/8192 CTA 下为 71.0/69.9/67.9us,三种请求数的峰值都落在同一总量),去掉它大 gather 退化 4.1x。公共签名不变,一个编译变体服务所有 launch 形状。每个 CTA 的逐行工作与原先完全一致,仅行→CTA 分配改变。此版本取代了初版(编译期 `clamp(ceil(max_rows/4), 1, 128)`)——初版使单请求 16k–262k 行的 prefill gather 最差慢 3.9x(275 vs 70us)。

## 关键改动

```text
vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py: grid.y 运行时按 max_rows/num_reqs 计算
tests/kernels/test_compressor_kv_cache.py: 新增 CuTeDSL grid 测试(全形状 + 尾部 + 强制网格)
```

```python
num_reqs = out.shape[0]
max_rows = out.shape[1] - offset
num_workers = cutlass.max(
    1, cutlass.min(cute.ceil_div(max_rows, 4), cute.ceil_div(8192, num_reqs))
)
grid = (num_reqs, num_workers, 1)
```

## 性能声明

(GB200;三臂各自捕获 CUDA graph 轮流重放 61 次取中位,block table 轮转 8 个区域避开 L2)

- BI decode 形状(本 PR 目标):32 请求 SWA(128 行)18.66→3.06us(6.10x);32 请求 compressed(~1k 行)22.06→10.80us(2.04x);规模越大收益越大,512 请求 SWA 283.74→19.28us(14.71x)、compressed 326.75→121.84us(2.68x);64/128 请求为 2.34–9.65x。全形状集无回退;prefill 侧单请求大 gather 1.06–1.09x,batched SWA 1.28x。
- 每 decode step(84 次调用,DSv4-Flash-Base,GB200 DP8/EP8,`VLLM_BATCH_INVARIANT=1`,sparse decode):gather 合计 1.66 → 0.51ms。
- 端到端:本 PR 是三个内核改动(#18、本 PR、#20)之一,合计使 BI=1 相对最优 BI=0 在 256 并发下的输出吞吐从 -10.8% 回到 +1.3%。

## 验证与回归风险

数值逐位一致:形状扫描每个 grid.y 下均 `torch.equal` 于旧内核与 Triton 参考;E2E 门禁(DSv4-Flash-Base,GB200 DP8/EP8,BI=1,FULL_DECODE_ONLY cudagraph,连同 #18)512/512 bitwise,run-to-run、batch-invariance 与 prefill/decode rescore 门禁均 bitwise。新增 `tests/kernels/test_compressor_kv_cache.py` CuTeDSL grid 测试覆盖 #42236 全形状、尾部(0/63/65/127/129 行,gather_len 0/1)与强制 1/3/1024 CTA 网格,GB200 上 25 passed。风险集中在 DSv4 NVIDIA 路径的 launch 几何;8192 平台值取自该硬件扫描,其他 GPU 未单独验证(PR 未注明)。

## 关联

#42236(引入该内核与 prefill 形状基准);#18 与 #20(同批次 BI decode 内核改动,三者合计带来 +1.3% 吞吐翻转);前序版本(编译期 clamp)已被本 PR 取代。pack 素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
