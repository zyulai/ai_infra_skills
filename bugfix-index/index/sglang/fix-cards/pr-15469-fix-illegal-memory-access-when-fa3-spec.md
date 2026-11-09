# PR #15469: Fix Illegal Memory Access when fa3 + spec + topk + page_size > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15469](https://github.com/sgl-project/sglang/pull/15469) |
| 作者 | yubofredwang |
| 合并日期 | 2025-12-23 |
| labels | run-ci |
| 热度 | 👍 6 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
FA3 + EAGLE 投机 + topk + page_size>1 时 page_table 计算漏加 last_partial_page 导致 IMA 崩溃;重写展开元数据构建修复。

## 问题现象
在 FlashAttention3 后端 + 投机解码(EAGLE)+ topk + page_size > 1 的组合下发生 CUDA Illegal Memory Access(IMA)崩溃。对应 issue #14624。基准数据(bench_sglang_eagle.py,80 questions):Throughput 1410.20 token/s,Acceptance length 2.67。正文未附崩溃日志原文。

## 根因
正文披露:此前的实现在计算 expand_metadata 的 page table 时,没有把 decode step pages 之前的第一个 last_partial_page 加进去,page_table 处理不正确,导致 IMA。正文还解释了为何不能像 PR #15107 那样用简单公式(`decode_length + 2*page_size - 1`)计算页数:out_cache_loc 分配到的页不一定是连续块(例如最后一个扩展页是 page 1,而 topk 分配到的首个页是 page 4),必须对每个 location 做 `// page_size` 再去重,并列举了 last_page_len 非 0/为 0、页不连续、decode 长度小于 page_size 等四类需要处理的情形。

## 修复方案
`python/sglang/srt/layers/attention/flashattention_backend.py`:
1. 将 `update_draft_decode_set_expand_metadata_with_page_size` 重写为 `draft_decode_set_expand_metadata`:对 `cache_loc // page_size` 后,用「变化点 mask + cumsum + scatter」的向量化 `unique_consecutive` 思路把去重后的页号搬到 page_table 前部,并以 `cache_seq_lens` 限制每页内 KV 的访问范围(正文所述方案);不再用减 last_page_len + strided index 的旧算法。
2. `init_cuda_graph_state` 中 expand 的 page_table 宽度由 `decode_length` 改为 `decode_length + 1`(为 last partial page 预留一列),删除 `strided_indices_expand` 缓存。
3. `init_forward_metadata` 非 CUDA graph 路径同样改用新函数并预分配 zeros 的 `expand_page_table`。
4. 测试:`test_flashattn_backend.py` 更新单测并新增 multi-batch/非连续页用例;eagle 推理测试引入 `page_size=256`(「Large page size tend to expose IMA bugs」)与 FA3 + page-size=256 的服务端测试;附带把 `benchmark/mtbench/bench_sglang_eagle.py` 的 `prompt` 字段改为 `turns`。

```diff
-    expand_page_table = cache_loc[:, :decode_length].clone()
-    ...
-    expand_page_table = (
-        expand_page_table[
-            :, strided_indices_expand[: (decode_length + page_size - 1) // page_size]
-        ]
-        // page_size
-    )
+    cache_loc = (cache_loc // page_size).to(torch.int32)
+    mask = torch.ones_like(cache_loc, dtype=torch.bool)
+    mask[:, 1:] = cache_loc[:, 1:] != cache_loc[:, :-1]
+    positions = mask.cumsum(dim=1) - 1
+    num_seqs = cache_loc.shape[0]
+    page_table[:num_seqs, :].scatter_(1, positions, cache_loc)
```

## 影响与回归风险
- 影响窗口:引入该 expand page table 旧算法的提交之后至本修复之前(正文未给具体版本)。
- 回归面:FA3 后端在投机解码 draft decode expand 阶段的 page table 构建(eager 与 CUDA graph 两条路径);page_table 形状多一列,attention kernel 读取该元数据的方式依赖 cache_seq_lens 截断;`torch.compile(dynamic=True)` 编译的新函数;相关单测与 EAGLE page_size 测试矩阵变化(page_size=64 用例被 256 取代)。

## 关联
- See: #14624(https://github.com/sgl-project/sglang/issues/14624)
- 对比方案来源: #15107(https://github.com/sgl-project/sglang/pull/15107)
- 其余:正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
