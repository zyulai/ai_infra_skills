# PR #1779: [Fix] Fix NaN issues by fixing the cuda graph padding values for flashinfer

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1779](https://github.com/sgl-project/sglang/pull/1779) |
| 作者 | merrymercy |
| 合并日期 | 2024-10-24 |
| labels | (none) |
| 热度 | 👍 5 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论

flashinfer CUDA graph replay 时 padding 请求的 seq_len 被填 0,在动态 fp8 量化模型上产生 NaN;改为填 1 修复。

## 问题现象

对 padded 请求,系统此前将 seq_len 设为 0,由此产生 NaN 问题(由 issue #1571 报告)。正文说明该问题大多只发生在使用动态 fp8 量化的模型(`--quantization fp8`)上;作者怀疑与动态量化和 flashinfer kernel 的组合有关(正文原话为 "I suspect",属推测)。影响面是启用 CUDA graph 且 batch 需要补齐(实际 batch size 小于 capture batch size)时的 flashinfer 推理。

## 根因

正文披露:CUDA graph replay 中,padding 请求的 `seq_lens` 用 `seq_len_fill_value`(即 0)填充;seq_len=0 的输入在 flashinfer kernel 与动态 fp8 量化组合下计算出 NaN。正文未给出更底层的 kernel 级机制,仅表达了与动态量化/flashinfer kernel 相关的怀疑;pack 中亦无进一步分析。

## 修复方案

将 replay 时 padding 请求的 seq_len 填充值从 0 改为 1,避免零长度序列进入 flashinfer kernel。

关键摘录(`python/sglang/srt/model_executor/cuda_graph_runner.py`,全部改动仅此 1 行):

```diff
         if bs != raw_bs:
-            self.seq_lens.fill_(self.seq_len_fill_value)
+            self.seq_lens.fill_(1)
             self.out_cache_loc.zero_()
```

`out_cache_loc.zero_()` 保持不变,padding 请求仍指向空位置,不产生真实输出。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,仅 `cuda_graph_runner.py` replay 中 batch size 补齐分支的 seq_len 填充值。影响所有走 CUDA graph replay 且 bs 不足的路径(不限于 fp8/flashinfer);seq_len 由 0 变 1 意味着 padding 请求会以长度 1 的假序列参与 kernel,正文未讨论对非 fp8 后端的副作用。改动一行、语义直观,风险低。

## 关联

- issue [#1571](https://github.com/sgl-project/sglang/issues/1571)(NaN 问题报告)
- 正文 cc 了 @yzh119(flashinfer 相关)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
