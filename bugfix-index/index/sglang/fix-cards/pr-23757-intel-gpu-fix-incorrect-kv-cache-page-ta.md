# PR #23757: [Intel GPU] Fix incorrect KV-cache page table for local attention when page_size > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23757](https://github.com/sgl-project/sglang/pull/23757) |
| 作者 | ckvermaAI |
| 合并日期 | 2026-05-26 |
| labels | intel, xpu, run-ci |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
XPU 后端 page_size>1 且启用 local attention 时,把 token 粒度的 req_to_token 表直接当 page 粒度块表传给 make_local_attention_virtual_batches,导致输出错误/全零;修复为先做 stride+除法归一化。

## 问题现象
在 XPU(Intel GPU)attention 后端,当 `page_size > 1` 且启用 local(分块)attention 时,传给 `make_local_attention_virtual_batches` 的 KV-cache page table 值不正确,导致 `flash_attn_with_kvcache` 输出错误或全零。触发条件是 page_size>1 与 local attention 同时启用。影响面上,正文给出的 XPU GSM8k 基准中修复前 Accuracy 仅 0.005(修复后 0.815)、Invalid 0.015(修复后 0.000),基本不可用。

## 根因
正文披露:`make_local_attention_virtual_batches` 期望的是 page 粒度的块表——第 p 列存逻辑页 p 对应的物理页索引;而原始 `req_to_token` 表是 token 粒度的(第 i 列 = token i 的 KV slot)。当 `page_size > 1` 时,把未做转换的 token 粒度表直接传入,使 `block_starts = k_seqstarts_absolute // page_size` 索引到错误的物理页值,产生错误的 KV-cache page table。

## 修复方案
修改 `python/sglang/srt/layers/attention/xpu_backend.py`:在 `page_size > 1` 时,先把 page table 转为 page 粒度再传给 `make_local_attention_virtual_batches`,转换分两步:(1) Striding——按 `torch.arange(0, ..., page_size)` 每隔 page_size 取一列;(2) Dividing——对取出的值整数除以 page_size,把 KV slot 索引换算成物理页索引。这与后端其他地方已对 `metadata.page_table` 做的归一化保持一致。关键 diff:

```python
+        if self.page_size > 1:
+            strided_indices = torch.arange(
+                0, page_table.shape[1], self.page_size, device=page_table.device
+            )
+            page_table = page_table[:, strided_indices] // self.page_size
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 XPU(Intel GPU)后端 `_init_local_attn_metadata` 中启用 local attention 的路径;page_size == 1 时不做转换、行为不变;page_size>1 时 page table 多一次 arange/切片/除法,有轻微额外开销,且此后传给 `make_local_attention_virtual_batches` 的表从 token 粒度变为 page 粒度。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
