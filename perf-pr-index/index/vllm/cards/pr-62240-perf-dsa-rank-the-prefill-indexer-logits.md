# PR #62240: [Perf][DSA] Rank the prefill indexer logits with DeepSelect in BF16

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Juntian777 |
| 合并 | 2026-10-28 |
| 分类 | Speculative Decoding(classify.py 给出;实际为 DSA 稀疏注意力 indexer 的 prefill top-k 优化) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62240 |

## 一句话结论

DSA prefill indexer 在 BF16 logits 上直接用 DeepSelect 排 top-k,免掉 FP32 上采,内核快 3.7x、prefill step 降 6.6%。

## 动机

`sparse_indexer_topk_backend="deep_select"` 下,decode 路径自 #56464 起已在 BF16 logits 上用 DeepSelect 取 top-k;prefill 路径却仍把 DeepGEMM 产出的 BF16 MQA logits 先 `as_fp32_logits` 上采成 FP32(占满双倍带宽与显存),再用 `top_k_per_row_prefill` 排序。upcast + FP32 top-k 成为 indexer 热点。#57206 曾尝试同样改动,但受限于当时 DeepGEMM 的 row_starts 布局只支持单请求 chunk 而被关闭;#61149 之后每个 chunk 行都从第 0 列开始,多请求 chunk 的因果区间可直接用 `end` 表达,限制解除。

## 优化点

在 `sparse_attn_indexer` 的 prefill 分支加守卫条件:`topk_backend == "deep_select"` 且 logits 为 bf16、`dcp_world_size == 1`、`is_deep_select_supported(logits, topk_tokens)` 时,跳过 `as_fp32_logits`,直接以 `deep_select_topk(logits, topk_tokens, end=row_ends, output_idx=topk_indices)` 在 BF16 上排序(利用 #61149 后行起点恒为 0 的布局,`end = cu_seqlen_ke - cu_seqlen_ks` 即每行完整因果区间);不满足条件(DCP、XPU、其他 top-k 后端、DeepSelect 读不了的行 stride)保持原 FP32 上采 + `top_k_per_row_prefill` 路径。数值上 BF16→FP32 是精确加宽,选出的 top-k 值与 `torch.topk` 完全一致,仅第 512 名并列时的索引集合可能不同——该平局行为此前(#56749)本就未定义。

## 关键改动

```text
vllm/model_executor/layers/sparse_attn_indexer.py: prefill 分支按守卫走 deep_select_topk(BF16)
```

```python
use_deep_select = (
    topk_backend == "deep_select"
    and logits.dtype == torch.bfloat16
    and dcp_world_size == 1
    and is_deep_select_supported(logits, topk_tokens)
)
if not use_deep_select:
    logits = as_fp32_logits(logits, row_ends)
```

## 性能声明

- Top-k 单独微基准(1x GB200,CUPTI,冷 L2,3 轮取中位;一个 query chunk 的行,对 65,536 个 compressed keys(128K V4.1 请求)带因果行尾,top-k 512,随机 BF16 logits;基线 `as_fp32_logits` + `top_k_per_row_prefill`):128 行 66.5→13.6us(4.9x),2048 行 414.4→110.1us(3.8x),4096 行 777.2→210.0us(3.7x),8192 行 1503.9→410.7us(3.7x)。
- E2E prefill step(`vllm bench forward`,DeepSeek-V4.1-Flash,dummy weights,DEP2 单 GB200 节点,`ds8x4k=8*4096@131072`,FlashMLA MegaAttn + MXFP4 indexer,rank 0 nsys):step 429.8→401.5ms(-6.6%);indexer logits+top-k 64.5→30.9ms,其中 upcast+top-k 39.4→5.7ms。
- 精度:GSM8K strict-match 两跑 0.9242/0.9227(±0.0073),passkey retrieval 在约 8K/31K/62K prompt 各 20/20;DSpark acceptance length 3.44(每 draft token 48.9%),与改动前同模型水平一致(0.9318 / 3.36,P/D 设置)。

## 验证与回归风险

复用既有测试,无新增:`pytest tests/kernels/test_top_k_per_row.py -k "deep_select or sparse_indexer"`、`test_mla_short_prefill_indexer.py`、`test_indexer_tp_row_shard.py`、`test_indexer_dcp_localize.py` —— 43 passed, 4 skipped(GB200)。既有 `test_deep_select_topk` 系列已覆盖依赖行为(BF16 输入、per-row end、短行 -1 填充、向宽 buffer 切片写输出)。风险面:仅 opt-in(`--kernel-config '{"sparse_indexer_topk_backend":"deep_select"}'`)生效,默认路径不变;守卫覆盖 dtype/DCP/支持性,未命中即回退 FP32。遗留:logits chunk 预算 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` 仍按 FP32 计,改按 BF16 计可使 chunk 高度翻倍,留作后续。

## 关联

#56464(decode 路径 DeepSelect,本 PR 将同一机制带到 prefill);#61149(DeepGEMM 压缩 logits 行布局,解锁多请求 chunk);#57206(被关闭的前次尝试);#62172(decode `cooperative_topk` 读 BF16);#60618(WIP,评分+top-k 融合新内核);#56749(既有 prefill 内核 tie-breaking)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
