# PR #60753: [Perf][ROCm][GLM-5.3-Flash] AITER prefill indexer topk

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simondanielsson |
| 合并 | 2026-10-14 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, rocm, ready, glm |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/60753 |

## 一句话结论

GLM-5.3-Flash 稀疏 indexer 的 prefill top-k 改用 AITER FlyDSL 内核,512k 上下文 TTFT 提升 20%+,索引器 topk 耗时降为约 1/3。

## 动机

GLM-5.3-Flash 的 sparse MLA 在 512k ISL 下,最后一个 prefill chunk 要把上下文按 kpool 切成 16 块,对每块独立跑 score + topk,且对 11 个稀疏 MLA 层各跑一遍:in-tree HIP kernel `top_k_per_row_prefill` 单次调用约 1.9 ms,合计 `1.9ms × 11 × 16 ≈ 350 ms`,而整个 forward 约 800 ms——chunked prefill 的 43% 时间花在 topk 上。AITER 的 FlyDSL 实现在该 shape 下明显更快(仅 gfx950 可用)。

## 优化点

调度层面替换 kernel:`sparse_attn_indexer_kpool` 中把 `torch.ops._C.top_k_per_row_prefill` 换成条件分派——`rocm_aiter_ops.is_indexer_top_k_supported(is_prefill=True, compress_ratio=index_kpool, num_rows=num_rows)` 通过时调用 AITER 的 `_launch_aiter_top_k_per_row_prefill`(FlyDSL 路径),否则回退原 in-tree kernel。注意收益来自 AITER 的 FlyDSL 变体而非其 HIP 内核:微基准显示强制 AITER 的 HIP one-block kernel 反而略慢(0.93×)、multi-block 更慢(0.45×)。附带新增 `benchmarks/kernels/benchmark_sparse_indexer_topk_prefill.py`,可对比 in-tree 与 AITER 路径并校验选中集合(逐行集合差)。

## 关键改动

```text
vllm/models/glm5next/amd/sparse_indexer.py: prefill topk 按平台能力分派到
  AITER FlyDSL,否则回退 torch.ops._C.top_k_per_row_prefill
benchmarks/kernels/benchmark_sparse_indexer_topk_prefill.py: 新增微基准脚本
```

关键机制摘录(sparse_indexer.py,分派条件):

```python
if rocm_aiter_ops.is_indexer_top_k_supported(
    is_prefill=True,
    compress_ratio=index_kpool,
    num_rows=num_rows,
):
    _launch_aiter_top_k_per_row_prefill(...)
else:
    torch.ops._C.top_k_per_row_prefill(...)
```

## 性能声明

E2E(`vllm serve zai-org/GLM-5.3-Flash` TP8,`ROCM_AITER_MLA_SPARSE` 后端,AITER sparse MLA/共享专家融合/breakable cudagraph 开启,`vllm bench serve` random 数据,OSL 1024):

- 512k ISL、并发 4:TTFT 22731.18 → 17798.95 ms(+21.70%),输出吞吐 52.58 → 64.30 tok/s(+22.29%);
- 256k ISL 并发 4 TTFT +15.64%;64k ISL 并发 8 TTFT +10.89%;8k ISL 并发 1/8 TTFT +12.88%/+12.24%;
- 1k ISL 基本中性(并发 8 TTFT -5.16%);ITL 全部 ±0.5% 内(符合预期);收益随上下文长度增长。

微基准(rows=1024, cols=131072, k=512, 512 MiB logits):FlyDSL auto 分派 non-causal 0.261 → 0.196 ms(1.33×),causal 0.161 → 0.113 ms(1.42×),选中行 0/1024 差异;逐 kernel 计时:topk 单次 1.9 ms → 694 µs(3×)。

## 验证与回归风险

精度:GSM8K 5-shot lm_eval,本分支 flexible 0.9174 vs main 0.9219(stderr ±0.0076,略低但在误差范围);GPQA-Diamond cot-zeroshot flexible 0.7020 vs main 0.6768。风险点:AITER 路径仅 gfx950 + 安装 AITER + `VLLM_ROCM_USE_AITER=1` 时可用,`is_indexer_top_k_supported` 不满足时回退原 kernel;收益与 kpool 压缩比、行数相关,小上下文(1k)有轻微回退;pr checklist 未勾选(作者未走 /pr-checklist 流程),测试主要靠 e2e sweep 与微基准而非单测。

## 关联

- 同区域:vLLM 索引中 ROCm AITER 系列优化(如 PR #59478 AITER FlyDSL int4 quickreduce、PR #39616 AITER MLA attention);GLM-5.3-Flash 模型支持见 PR #53906
- 相邻:PR #62172(cooperative_topk 读 BF16 logits,NVIDIA 侧的 indexer top-k 优化,思路相对)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
