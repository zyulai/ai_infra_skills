# PR #10214: [Fix] KV-cache eviction mismatch across PP ranks in DeepSeek V3/R1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10214](https://github.com/sgl-project/sglang/pull/10214) |
| 作者 | qhsc |
| 合并日期 | 2025-09-09 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

PP 各 rank KV cache 容量不一致致不对称逐出、prefix 缓存状态分歧崩溃;跨 rank all_reduce MIN 取最小值修复。

## 问题现象

PP=2 部署 DeepSeek V3/R1 模型时,pp0 与 pp1 为 KV cache 预留的最大总 token 数(max_total_num_tokens)不相等。开启 radix cache 后运行一段时间,由于不对称的 cache 逐出(asymmetric cache evict),同一请求的 prefix cached token 数在各 PP rank 之间变得不一致,最终导致系统崩溃。正文附有截图说明 KV cache 大小不一致的现象。

## 根因

正文披露:不同 PP rank 的层数不同(DeepSeek V3/R1 结构下),导致 `init_memory_pool` 计算出的 `max_total_num_tokens` 在各 rank 上不相等;radix cache 的容量上限随 rank 不同,逐出行为不对称,运行一段时间后同一请求在各 PP rank 上的 prefix 缓存命中状态失去一致性,进而引发崩溃。

## 修复方案

在 `python/sglang/srt/model_executor/model_runner.py` 的 `init_memory_pool` 中,当 `pp_size > 1` 时,把 `max_total_num_tokens` 包成 int64 张量,通过 `torch.distributed.all_reduce`(ReduceOp.MIN,`get_world_group().cpu_group`)跨 PP rank 取最小值,使所有 rank 使用相同的 KV cache 容量上限。

关键 diff 摘录:

```diff
+        # different pp rank may have different num of layers, so we need to reduce the max_total_num_tokens
+        if self.pp_size > 1:
+            tensor = torch.tensor(self.max_total_num_tokens, dtype=torch.int64)
+            torch.distributed.all_reduce(
+                tensor,
+                op=torch.distributed.ReduceOp.MIN,
+                group=get_world_group().cpu_group,
+            )
+            self.max_total_num_tokens = tensor.item()
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅影响 `pp_size > 1` 时的内存池初始化:各 rank 的 `max_total_num_tokens` 统一为最小值,意味着较大容量 rank 的可用 KV cache 容量被下调,显存利用率略降但换来跨 rank 状态一致;PP=1 路径与 hybrid cache 分支(`set_num_token_hybrid` 在取 MIN 之后执行)不受影响。正文未提供 accuracy 测试结果(Checklist 均未勾选)。

## 关联

正文未列出 issue/PR 链接(仅 @ 提及协作者 @jianfei-wangg)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
