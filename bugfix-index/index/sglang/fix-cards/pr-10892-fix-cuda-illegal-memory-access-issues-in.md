# PR #10892: Fix CUDA illegal memory access issues in speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10892](https://github.com/sgl-project/sglang/pull/10892) |
| 作者 | ur4t |
| 合并日期 | 2025-10-03 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE CUDA graph runner 重放时 bs 与捕获 bs 不一致导致 IMA,replay 中补上 `self.positions.zero_()` 清零残留 positions。

## 问题现象
EAGLE CUDA graph runner 中出现 CUDA illegal memory access(IMA)类崩溃,关联 issue:#6309、#8296、#8336、#9888。正文未附具体报错日志;作者说明本 PR 旨在修复 IMA 的根因,区别于 #9687 用 padding 方式缓解。

## 根因
正文披露的修复思路是:当 `raw_bs` 与 `capture_bs` 不一致(重放尺寸小于捕获尺寸、存在 padding 槽位)时,`self.positions` 缓冲区未被重置——修改项写明 "Reset `self.positions` in EAGLE CUDA graph runners when `raw_bs` and `capture_bs` differ."。即重放时对 padded 部分虽然清了 `seq_lens` 与 `out_cache_loc`,但 positions 里残留上一次更大 batch 的位置值,导致后续按 positions 寻址时越界,产生 IMA。

(由 diff 推断)两个 runner 的 `replay` 中原本已有 `self.seq_lens.fill_(self.seq_len_fill_value)` 和 `self.out_cache_loc.zero_()` 的 padding 清理,唯独漏了 `positions`;残留的非零 position 使内核对 KV cache/嵌入等按位置索引的缓冲区访问到无效地址。

## 修复方案
在两个 EAGLE CUDA graph runner 的 `replay` 中,bs padding 清理块内增加一行 `self.positions.zero_()`:

1. `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`(条件 `if bs != raw_bs:`)
2. `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`(条件 `if bs * self.num_tokens_per_bs != num_tokens:`,其后原有 `accept_length.fill_(1)`、`extend_seq_lens.fill_(1)`)

关键 diff 摘录:

```diff
         if bs != raw_bs:
             self.seq_lens.fill_(self.seq_len_fill_value)
             self.out_cache_loc.zero_()
+            self.positions.zero_()
```

改动为两处各一行,与 #9687 的 padding 缓解方案不同,直接消除越界来源。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 EAGLE draft 与 draft-extend 两个 CUDA graph runner 的 `replay` 路径,且只在实际 bs 小于捕获 bs 的 padding 情况下多做一次 positions 清零;正常尺寸重放与捕获逻辑不变。positions 被清零后 padded 槽位按位置 0 参与 padding 计算(与 seq_lens fill 值、out_cache_loc 清零配合),对真实 token 的输出无影响;其他推测解码运行路径不受影响。

## 关联
正文列出的相关 issue/PR:
- #6309
- #8296
- #8336
- #9888
- #9687(padding 缓解方案,与本 PR 对比)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
