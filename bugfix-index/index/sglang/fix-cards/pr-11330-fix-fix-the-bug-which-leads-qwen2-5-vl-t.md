# PR #11330: fix: fix the bug which leads qwen2_5_vl to crash with mixed_chunk

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11330](https://github.com/sgl-project/sglang/pull/11330) |
| 作者 | PanJason |
| 合并日期 | 2025-11-03 |
| labels | run-ci |
| 热度 | 👍 6 · 💬 3 |
| 分类 | Model-specific Fast Path |
| bug 类型 | crash |

## 一句话结论
mixed_chunk 混批下 decode 请求拿到空 mrope_positions 致 qwen2_5_vl 崩溃,现改用 mrope_position_deltas 补齐。

## 问题现象
开启 mixed_chunk 时,qwen2_5_vl 在对 query 和 key 做 reshape 时崩溃,现象与 #10623 描述一致。触发条件:一个 batch 同时混有 prefill 与 decode 请求(mixed batch)。正文未描述具体报错日志,崩溃发生在 reshape query/key 处。

## 根因
正文披露:query/key 与 position 不匹配。若 batch 是混合的,`mm_input` 中的 `mrope_positions` 只为 prefill 做了准备,导致同批中的 decode 请求拿到空的 position 向量,进而 query、key 与 positions 之间维度不匹配,reshape 时崩溃。

## 修复方案
改 `python/sglang/srt/model_executor/forward_batch_info.py`:
1. 新增辅助函数 `_expand_mrope_from_input`,基于 `mrope_position_delta` 为 decode 请求构造 mrope_positions(即正文所说"use mrope_position_deltas for decode requests in a mixed batch"):

```diff
+    def _expand_mrope_from_input(
+        self,
+        mm_input: MultimodalInputs,
+        seq_len: int,
+        device: torch.device,
+    ) -> torch.Tensor:
+        ...
+        mrope_position_deltas = mm_input.mrope_position_delta.flatten()
+        mrope_positions = (
+            (mrope_position_deltas + seq_len - 1).unsqueeze(0).repeat(3, 1)
+        )
+        return mrope_positions
```

2. 原纯 decode 分支中的 delta 搬运/展开逻辑收敛为调用该辅助函数(含首次运行时 host-to-device 搬运逻辑)。
3. extend 分支:当切出的 `mrope_positions.numel() == 0`(混批中的 decode 请求)时,回退调用 `_expand_mrope_from_input` 用 delta 补齐 position。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;问题在 #10623 中报告。
- 回归面:`forward_batch_info.py` 的 `_compute_mrope_positions` 是所有 mrope 模型(qwen2-vl/qwen2.5-vl 系)position 计算的公共路径;decode 分支逻辑等价重构,extend 分支仅在 numel==0 时新增回退,理论上仅影响 mixed_chunk 下混批场景,纯 prefill/纯 decode 路径行为不变。该 PR 未按 checklist 提供 accuracy/benchmark 数据。

## 关联
- [#10623](https://github.com/sgl-project/sglang/issues/10623)(崩溃现象报告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
