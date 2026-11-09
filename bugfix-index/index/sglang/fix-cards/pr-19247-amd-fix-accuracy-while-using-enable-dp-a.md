# PR #19247: [AMD] Fix accuracy while using --enable-dp-attention

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19247](https://github.com/sgl-project/sglang/pull/19247) |
| 作者 | yichiche |
| 合并日期 | 2026-02-25 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
--enable-dp-attention(tp8/dp8)下每卡 128 头,AMD persistent MLA decode kernel 被用在不支持的 head 数上致精度问题;把 128 头也排除出该 kernel。

## 问题现象
`--enable-dp-attention --tp-size 8 --dp-size 8` 时 attention TP size 变为 1(8//8),每张 GPU 看到全部 128 个 head 而非 16,导致 persistent MLA decode kernel 被错误选用于它不支持的 head 数,出现精度问题。修复后正文实测:Accuracy 0.951,Invalid 0.000,Latency 69.849 s,Output throughput 1878.389 token/s。

## 根因
正文披露:aiter 后端选择 persistent kernel 的条件只排除了 `num_head == 16` 的场景;代码注释里提到 persistent kernel 不支持 head-size=128,但条件中没有包含该情况,于是 tp8/dp8(128 头)落入了 persistent kernel 的使用范围。

## 修复方案
`python/sglang/srt/layers/attention/aiter_backend.py`:在 fp8 kv_cache 之外的判断里,把 128 头也归入回退 non-persist kernel 的条件:

```python
-            if self.num_head == 16 and self.kv_cache_dtype is not fp8_dtype:
+            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
+            if (
+                self.num_head == 16 or self.num_head == 128
+            ) and self.kv_cache_dtype is not fp8_dtype:
                 _use_mla_ps_kernel = False
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:仅 AMD aiter 后端 persistent MLA kernel 的选择条件;16 头与 128 头且非 fp8 kv_cache 时改走 non-persist(与修复前 16 头行为一致),fp8 kv_cache 下仍可用 persistent kernel。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
