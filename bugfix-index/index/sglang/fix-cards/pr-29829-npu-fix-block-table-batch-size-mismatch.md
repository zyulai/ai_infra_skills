# PR #29829: [NPU] Fix block_table batch size mismatch in GLM-4.7-Flash DeepEP + MTP without CUDA Graphs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29829](https://github.com/sgl-project/sglang/pull/29829) |
| 作者 | Estrella-xx |
| 合并日期 | 2026-07-02 |
| labels | npu, run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
NPU 禁用 CUDA Graph 时 DeepEP+MTP 使 block_table 与 query 的 batch size 不匹配;forward_mtp 将 block_table/actual_seq_lengths_kv 切到实际 batch size。

## 问题现象
GLM-4.7-Flash 上,禁用 CUDA Graphs 且同时启用 DeepEP 与 MTP 时,block table 与 query 之间出现 batch size 不匹配。正文未给出具体报错日志,仅以 accuracy 测试截图佐证修复有效;影响面为 NPU + 非 graph 模式 + DeepEP + MTP 组合。

## 根因
正文 Modifications 说明修复为"将 block table 与 actual_seq_lengths_kv 切到实际 batch size,使其第一维与 query batch size 一致";其机制在 diff 注释中解释(由 diff 推断):非 graph_mode 下 query 被 slice 到 `num_token_non_padded`(可能丢弃已完成请求),而 block_table 未同步切片;FIA TND kernel 要求 `block_table.shape[0] == len(actual_seq_lengths)`,不满足即错配。

## 修复方案
`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py::forward_mtp`:非 graph_mode 时按实际 batch 切片,并传给两处 FIA 调用(替换原来的整表 `self.forward_metadata.block_tables`):

```python
+            if not self.graph_mode:
+                actual_bs = len(actual_seq_lengths)
+                block_table = self.forward_metadata.block_tables[:actual_bs]
+                actual_seq_lengths_kv = actual_seq_lengths_kv[:actual_bs]
+            else:
+                block_table = self.forward_metadata.block_tables
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:仅 NPU ascend 后端 `forward_mtp` 的元数据准备(graph_mode 保持整表);两处 FIA kernel 调用的 block_table 实参随切片变化,非 DeepEP/MTP 的路径不受影响。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
