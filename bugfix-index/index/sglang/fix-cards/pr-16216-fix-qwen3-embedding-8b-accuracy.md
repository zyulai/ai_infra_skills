# PR #16216: fix qwen3-embedding-8b accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16216](https://github.com/sgl-project/sglang/pull/16216) |
| 作者 | amote-i |
| 合并日期 | 2025-12-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
qwen3-embedding-8b 精度问题:Qwen3 前向中 embedding 模型在 NPU 上走了非 native QKV 路径;改为 embedding 模型一律走 `forward_prepare_native`。

## 问题现象
正文未描述具体现象——Motivation / Modifications / Accuracy Tests 等小节均为模板占位未填写。按标题:qwen3-embedding-8b 模型存在精度(accuracy)问题。

## 根因
正文未披露根因。(由 diff 推断)`python/sglang/srt/models/qwen3.py` 的 attention 前向中,进入 `forward_prepare_native` 的条件是 `not _is_npu`,即 NPU 上 embedding 模型走的是另一条(QKV 准备)路径,该路径对 embedding 模型产生精度问题;修复让 `is_embedding` 场景即使 `_is_npu` 为真也强制走 native 路径。

## 修复方案
`python/sglang/srt/models/qwen3.py` 一行条件修改:

```python
-        if not _is_npu:
+        if not _is_npu or get_global_server_args().is_embedding:
             q, k, v = self.forward_prepare_native(
                 positions=positions,
                 hidden_states=hidden_states,
```

即 embedding 模型(`get_global_server_args().is_embedding` 为真)不再受 `_is_npu` 分支限制,统一使用 `forward_prepare_native` 准备 QKV。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Qwen3 模型 attention 前向的 QKV 准备路径选择;仅 `is_embedding` 为真的部署在 NPU 上改变路径,非 embedding 模型与非 NPU 平台的条件结果不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
