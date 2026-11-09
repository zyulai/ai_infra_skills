# PR #46928: [Fix] [AMD] aiter: Fix garbage decode output with unified attention at page size 1 under CUDA graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46928](https://github.com/sgl-project/sglang/pull/46928) |
| 作者 | jiaqiang-dot-liu |
| 合并日期 | 2026-11-05 |
| labels | run-ci, run-ci-amd-extra |
| 热度 | 👍 0 · 💬 16 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
AMD aiter unified attention 在 page size 1 + CUDA graph 下 decode 页表没被拷贝、读到全零页表输出乱码;把拷贝条件放宽为所有非 SWA 配置。

## 问题现象
`SGLANG_USE_AITER_UNIFIED_ATTN=1`、dense(非 sliding-window)模型、默认 page size 1、CUDA graph 开启时,decode 输出乱码:prefill 产生的第一个 token 正确,之后每个 decode token 都错。greedy 采样下 `"The capital of France is"` 会续写成 `" a ( ( ( ( ( ..."` 而不是正确内容,GSM8K 掉到 0.00。没有任何配置校验拒绝这个组合,问题静默失败。Sliding-window 模型和 Qwen3-VL(model override 给了 page size 16)不受影响;非 CUDA graph 路径直接构建页表,也不受影响。

## 根因
正文披露:`_apply_cuda_graph_metadata` 中,unified-attention 页表只在 sliding-window pool 或 `page_size > 1` 时才拷贝进 captured 的 `cuda_graph_page_table`;page size 1 时什么都不拷贝,`unified_attention` 读到一张全零页表。而 `_transform_table_1_to_real` 在 page size 1 时本就是恒等变换,所以 `page_size > 1` 这个条件只是白白跳过了拷贝。读同一页表的另一条 decode 路径 SHUFFLE 5D KV layout 要求 page size 能被 8(BF16 KV)或 16(FP8 KV)整除,永远不会跑在 page size 1;AMD CI 测试与 cookbook 配方开启 unified attention 时都设了 `--page-size 16` 或 `64`,page size 1 此前没有测试覆盖。

## 修复方案
`python/sglang/srt/layers/attention/aiter_backend.py` 中把拷贝页表的分支条件从 `elif self.page_size > 1:` 改为 `else:`,即对所有非 sliding-window 配置都拷贝页表;并新增 AMD CI 测试 `test/registered/attention/test_aiter_unified_attention_page_size_1.py`(dense 模型 + unified attention + page size 1,GSM8K 阈值 0.65)。该测试在 main 上失败(0.0),打上补丁后 MI355X 0.765、MI300X 0.765。

```diff
-                        elif self.page_size > 1:
+                        else:
                             page_indices = self._transform_table_1_to_real(page_indices)
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:aiter backend CUDA graph 元数据应用路径,所有非 SWA 配置现在都会执行页表拷贝(page size > 1 行为与之前一致,page size 1 从"不拷贝"变为"拷贝");实测 page size 1 输出吞吐与 page size 16 相差 2% 以内,其余配置速度无变化。

## 关联
正文未列出 issue/PR 链接(注明由 [Hyperloom](https://github.com/AMD-AGI/Hyperloom) 发现)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
