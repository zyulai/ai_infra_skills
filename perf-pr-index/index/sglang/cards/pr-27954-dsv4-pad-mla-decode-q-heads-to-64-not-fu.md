# PR #27954: [dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-06-16 |
| 分类 | Attention / Kernel(tier: major) |
| Label | high priority, deepseek, run-ci, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 全量保留(单文件 deepseek_v4.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/27954 |

## 一句话结论

DSV4 MLA decode 在 attention-TP 下把 per-rank q-heads pad 到 64 而非 128,命中 FlashMLA 的 decode::head64 kernel,单 kernel 快约 3.4×,prefill 吞吐 +7.0%。

## 动机

DeepSeek-V4-Pro 在 tensor parallelism 下 MLA decode attention 一直走 FlashMLA 较慢的 `fwd_for_small_topk::head128` kernel:attention-TP 时每 rank 的 query head 数是 `n_heads // attn_tp`(128 / TP4 = 32),但 `MQALayer.forward` 在进 kernel 前把它 pad 到全局 `n_heads`(128),导致每 rank 白算 `n_heads - n_local` 个浪费的 head。FlashMLA 只对 `h_q ∈ {64, 128}` 做特化,pad 到 64 即可派发约 2 倍便宜的 `decode::head64` 变体。

## 优化点

`models/deepseek_v4.py` 的 `MQALayer.forward`:当 `n_local_heads <= 64` 时把 padded query buffer 尺寸设为 64(而非 `n_heads`),本 rank 的 heads 放在 `[:n_local_heads]`,输出后 `o[:, :n_local_heads, :]` 切回。`attn_sink` 按本 rank 切片后 pad 到同样尺寸;由于它是常量,首次 forward 构建一次并缓存(`_attn_sink_local`)——逐次重建会在 decode CUDA graph 里每层重放一次 fill+copy。unified-kv-triton 路径不动。

## 关键改动

```text
python/sglang/srt/models/deepseek_v4.py:
  padded_num_heads = 64 if self.n_local_heads <= 64 else self.n_heads
  sink 缓存于 self._attn_sink_local(首次 forward 构建)
  FlashMLA 调用改传 _attn_sink_local
```

## 性能声明

Kernel 级(DSV4-Pro fp4,B300 TP4):MLA decode attention 派发从 `fwd_for_small_topk::head128` 58.1 µs/launch → `decode::head64` 17.1 µs/launch,约 3.4×。

端到端(同硬件,ISL 8192/OSL 1024,fp8 KV cache,mixed-chunk,piecewise CUDA graph,swa-full-tokens-ratio 0.15,chunked-prefill 8192,sa-bench 并发扫描,中位数):Total TPS/GPU conc8 946→967(+2.2%)、conc16 1456→1507(+3.5%)、conc32 2116→2218(+4.8%)、conc64 2795→2980(+6.6%)。增益来源分析(running-req ~32):prefill input throughput 13,027→13,941 tok/s(+7.0%)、prefill per-step 571→545 ms(-4.6%),decode iter time 20.30→20.35 ms(≈0)——收益落在 prefill(extend 路径在小 chunk 时也派发该 decode kernel 且在关键路径上),decode attention 不在关键路径。

精度:GSM8K(1319,5-shot,chat,max_tokens=16384,DSV4-Pro fp4,B300 TP4)0.962 → 0.963。

## 验证与回归风险

仅上述 GSM8K 精度对比与 sa-bench 吞吐数据;checklist 的单测/精度项未勾选,CI Base 为红(Extra 通过)。风险面:仅当 `n_local_heads <= 64` 生效,`n_local_heads > 64`(低 TP)行为不变;依赖 FlashMLA 对 h_q {64,128} 的特化假设,FlashMLA 版本变化可能影响派发;padded heads 的输出被切掉,依赖 kernel 对 pad 位置的输出无副作用;`_attn_sink_local` 首建时机假设权重已加载。

## 关联

同区域:PR #23882(DeepSeek V4 主支持)、PR #38440(DSpark compact mode)、PR #24880(HiSparse DSV4);同分类 Attention / Kernel 其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
