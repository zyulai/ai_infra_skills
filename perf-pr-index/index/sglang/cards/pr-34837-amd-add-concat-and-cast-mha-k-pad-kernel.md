# PR #34837: [AMD] Add concat_and_cast_mha_k_pad_kernel to support 12-head and enable K3 aiter prefill kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-08-15 |
| 分类 | Attention / Kernel |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 统计;diff 内容本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/34837 |

## 一句话结论

新增非 2 幂 head 数的 K 拼接 kernel,让 Kimi-K3(TP8 每 GPU 12 heads)走 AITER MLA prefill:TTFT 最高提升 14.9%、吞吐最高 4.6%。

## 动机

MLA prefill 上 Triton 的 `extend_attention_fwd` 远慢于 aiter 内核(aiter 的 `_fmha_fwd_bf16_opus_fwd`,底层 `gqa_d192_v128_kernel`)。启用 aiter prefill 需要 `concat_and_cast_mha_k_kernel` 做 K 的 nope/rope 拼接与 cast,但该 kernel 用 `tl.arange(0, head_cnt)`,要求 head 数为 2 的幂;Kimi-K3 在 TP8 下每 GPU 恰好 12 heads,触发 "arange's range must be a power of 2" 崩溃,K3 因此用不上 AITER prefill kernel。同时 FP8 prefill attention kernel 不支持 12 heads,需 `SGLANG_AITER_FP8_PREFILL_ATTN=0` 走 bf16 路径。

## 优化点

在 `cache_ops.py` 新增 `concat_and_cast_mha_k_pad_kernel`:head 维按 `HEAD_BLOCK = next_power_of_2(head_cnt)` 展开、尾部掩码,src nope 块按 head 掩码搬运,rope 部分(所有 head 共享同一行)load 一次再 broadcast 写到各 head;`concat_and_cast_mha_k_triton` 在 `head_cnt != next_power_of_2(head_cnt)` 时分派到新 kernel(注释点明 Kimi-K3 96 heads 在 TP8 得 12)。原 kernel 与所有调用点不变,2 幂 head 数模型(如 DeepSeek 家族的 16/32 heads)走原路径零影响。这个 kernel 本身不是加速点,它是解锁 aiter `gqa_d192_v128_kernel` prefill 路径的使能器,收益来自 Triton extend attention → aiter attention 的替换。

## 关键改动

```text
python/sglang/kernels/ops/kvcache/cache_ops.py: 新增 concat_and_cast_mha_k_pad_kernel + 非 2 幂 head 分派
```

关键片段(head 维掩码填充):

```python
head_range = tl.arange(0, HEAD_BLOCK)   # HEAD_BLOCK == next_power_of_2(head_cnt)
head_mask = head_range < head_cnt
...
src_rope = tl.load(src_rope_ptr)
tl.store(dst_rope_ptr,
         tl.broadcast_to(src_rope, (HEAD_BLOCK, rope_dim)),
         mask=rope_mask)
```

## 性能声明

PR 正文:"With this change, TTFT improves by up to 14.9% and throughput by up to 4.6%, with larger gains at higher concurrency"。测试条件:Kimi-K3,8k1k(ISL 8192/OSL 1024),并发 2-64,TP8,完整 AITER 环境组合(`SGLANG_USE_AITER=1`、`SGLANG_AITER_K3_OPT=1`、`--prefill-attention-backend aiter` 等),sglang 为作者 opt 分支 + #34741、aiter 为 integration/k3-core-only 分支;逐并发数字在正文的截图里,文本未列出。精度:GSM8K 1319 题 Accuracy 0.951 / Invalid 0.001 / 输出吞吐 1179.339 token/s(单次运行,未给 baseline 对照)。

## 验证与回归风险

无新增单测(diff 仅含 kernel 与分派,checklist 未勾选测试项)。风险面低:仅当 head 数非 2 幂时才进新 kernel,原路径逐字节不变;rope broadcast 语义与原 kernel 相同(每 head 复制同一 rope 行)。注意:该收益依赖一整套未进 main 的分支与 AITER 环境(如 `SGLANG_AITER_FP8_PREFILL_ATTN=0`、K3 专用 opt 分支),12 heads + FP8 prefill 的组合仍不可用;PR CI:Base 通过,Extra 失败(x)。

## 关联

AMD + MLA attention:PR #37889(GLM DSA prefill top-k v2,同为 ROCm prefill 内核使能)、PR #32340(DSV4 shared experts fusion)。同 head 数问题模式:PR #32340 的 tl.arange 非 2 幂修复(top-6)与本 PR(12 heads)同源。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
