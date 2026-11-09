# PR #19070: fix(dense): fix Qwen3.5 dense model precision bug in TP_SIZE>1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19070](https://github.com/sgl-project/sglang/pull/19070) |
| 作者 | zju-stu-lizheng |
| 合并日期 | 2026-02-25 |
| labels | run-ci |
| 热度 | 👍 4 · 💬 2 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Qwen3.5 dense 块未接入 allreduce fusion 通信管理导致 TP>1 精度错误,按 MoE/dense 分流并在融合时跳过显式通信修复。

## 问题现象
正文未描述具体现象(无报错、复现步骤或数值对比)。从标题可知:Qwen3.5 dense 模型在 TP_SIZE>1 时出现精度 bug。

## 根因
正文未披露根因,正文只列出了修改要点。由 diff 推断:`Qwen3.5DecoderLayer.forward` 原先对 MLP 统一调用 `self.mlp(hidden_states, forward_batch, use_reduce_scatter)` 后无条件执行 `postprocess_layer`(显式通信);而 MoE 稀疏块与 dense MLP 块的前向签名与 allreduce fusion 接入方式不同,dense 路径既未参与 `should_fuse_mlp_allreduce_with_next_layer` 的融合判定,也未向下游标记 `_sglang_needs_allreduce_fusion`,在启用 allreduce fusion 且 TP>1 时通信与融合语义不一致,产生精度错误。

## 修复方案
改动位于 `python/sglang/srt/models/qwen3_5.py` 的两处 decoder layer `forward`(第 400 行与第 633 行附近的两个类,改动相同):
- 调用 `self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)` 得到 `should_allreduce_fusion`;
- MoE 块(`Qwen2MoeSparseMoeBlock` 实例)保持原调用 `(hidden_states, forward_batch, use_reduce_scatter)`;
- dense MLP 改为 `(hidden_states, should_allreduce_fusion, use_reduce_scatter)` 三参调用;
- 融合启用时给 `hidden_states` 打上 `hidden_states._sglang_needs_allreduce_fusion = True` 标志并跳过 `postprocess_layer` 显式通信;未融合时保持原 `postprocess_layer` 流程。

关键 diff 摘录:

```python
if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
    hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)
else:
    hidden_states = self.mlp(
        hidden_states, should_allreduce_fusion, use_reduce_scatter
    )
if should_allreduce_fusion:
    hidden_states._sglang_needs_allreduce_fusion = True
else:
    hidden_states, residual = self.layer_communicator.postprocess_layer(
        hidden_states, residual, forward_batch
    )
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:仅 `qwen3_5.py` 内两个 decoder layer 的 MLP 后处理路径。MoE 分支调用序列不变;dense 分支在关闭 allreduce fusion 时仍走原 `postprocess_layer`,行为等价;开启 fusion 时跳过显式通信、依赖下游按 `_sglang_needs_allreduce_fusion` 标志接管通信(由 diff 推断)。`forward_batch` 与 `use_reduce_scatter` 的原语义保留(正文亦如此声明)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
