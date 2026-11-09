# PR #28025: [BugFix] Fix incorrect preallocated sampled_token_ids tensor size

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28025](https://github.com/vllm-project/vllm/pull/28025) |
| 作者 | njhill |
| 合并日期 | 2025-11-04 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
V1 GPU model runner 预分配的 sampled_token_ids pinned CPU 缓冲误用 max_model_len 定尺寸,远超实际需求数;改为按 max_num_reqs 分配。

## 问题现象
正文未描述具体现象。从改动看,受影响的是 `vllm/v1/worker/gpu_model_runner.py` 中 `sampled_token_ids_pinned_cpu` 这块 pinned CPU 缓冲的预分配大小,属资源分配量级错误(由 diff 推断)。

## 根因
正文未披露根因。由 diff 推断:`__init__` 中该缓冲以 `(self.max_model_len, 1)` 预分配——即按单条序列的最大长度定容,而每步采样每条请求只需一个 token id,容量应取决于并发请求数 `max_num_reqs`;用 `max_model_len` 会在长上下文配置下把这块 pinned memory 放大数个量级。

## 修复方案
改 `vllm/v1/worker/gpu_model_runner.py` 的预分配形状,一行修正:

```diff
         self.sampled_token_ids_pinned_cpu = torch.empty(
-            (self.max_model_len, 1),
+            (self.max_num_reqs, 1),
             dtype=torch.int64,
             device="cpu",
             pin_memory=self.pin_memory,
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅改动 V1 GPU model runner 中这一处缓冲分配,降低 pinned CPU 内存占用;下游写入该缓冲的逻辑需与 `max_num_reqs` 维度匹配,若还有按 `max_model_len` 索引/拷贝的旧假设则可能越界(由 diff 推断,diff 内未见其他改动)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
