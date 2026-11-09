# PR #19760: Fix dp_attention crash when dp_size < tp_size in warmup dummy run

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19760](https://github.com/sgl-project/sglang/pull/19760) |
| 作者 | yhyang201 |
| 合并日期 | 2026-03-04 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
`--enable-dp-attention` 且 dp_size < tp_size(如 `--tp 8 --dp 4`)时,`_dummy_run` warmup 未按 attn_tp_size 对齐 num_tokens,启动即报 ValueError;补上对齐修复。

## 问题现象
以 `--enable-dp-attention` 启动且 dp_size < tp_size(正文示例 `--tp 8, --dp 4`)时,启动阶段(warmup dummy run)直接抛出:

```
ValueError: input tensor must be the same size as output size times world size
```

即服务尚未开始推理就在 warmup 阶段崩溃,影响所有该并行组合的启动。

## 根因
正文披露:`_dummy_run` warmup 路径缺少对 num_tokens 的 attn_tp_size 对齐处理;而同样的对齐在推理路径(`forward_batch_info.py`)和 `cuda_graph_runner.py` 中已经做了。warmup 时 num_tokens 不是 attn_tp_size 的倍数,collective 操作中输入张量与「output size × world size」不一致,触发上述 ValueError。

## 修复方案
改 `python/sglang/srt/model_executor/model_runner.py` 的 `_dummy_run`:在 `num_tokens = batch_size * num_tokens_per_bs` 之后,当 `require_gathered_buffer(self.server_args)` 成立且 `attn_tp_size > 1` 时,把 num_tokens 向下取整到 attn_tp_size 的倍数,并同步回调 batch_size:

```diff
+        if require_gathered_buffer(self.server_args):
+            attn_tp_size = get_attention_tp_size()
+            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
+                num_tokens = num_tokens // attn_tp_size * attn_tp_size
+                batch_size = num_tokens // num_tokens_per_bs
```

同时从 `sglang.srt.layers.dp_attention` 导入 `get_attention_tp_size`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;受影响配置为 `--enable-dp-attention` 且 dp_size < tp_size。
- 回归面:仅 `_dummy_run` warmup 路径,dummy batch 的 token 数在不能整除 attn_tp_size 时会被略减(最多减 attn_tp_size-1 个 token),不影响真实推理;attn_tp_size == 1 或未开 gathered buffer 时行为完全不变。与推理路径、CUDA graph runner 已有的对齐逻辑保持一致。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
