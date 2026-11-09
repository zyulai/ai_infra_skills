# PR #44408: Fix MiDashengLM TP>1 crash in audio encoder attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44408](https://github.com/vllm-project/vllm/pull/44408) |
| 作者 | mganczarenko |
| 合并日期 | 2026-06-09 |
| labels | ready, verified |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
DashengAttention.forward 用完整 embed_dim 做 reshape 而非 TP 感知的 head_dim/q_size,TP>1 时形状不匹配崩溃,改用分片尺寸修复。

## 问题现象
`mispeech/midashenglm-7b` 音频编码器在 `tensor_parallel_size > 1` 时首次推理请求即崩,服务不可用。修复前 TP=2 报错:
```
RuntimeError: shape '[1, 1568, 3, 8, 64]' is invalid for input of size 602112
```
Server crashes during first inference request。TP=1 不受影响(修复对非分片场景是 no-op)。

## 根因
正文披露根因:`DashengAttention.forward()` 在张量 reshape 时使用完整的 `embed_dim`(即局部变量 `C`)而不是 TP 感知的 `self.head_dim` 和 `self.q_size`。当 `tensor_parallel_size > 1` 时,每张卡的 QKV 投影输出已是按 TP 分片后的尺寸(如 3 heads × 64 dim),而 reshape 仍按完整模型的 embed_dim 形状(`3, 8, 64`,即 8 heads)解释,元素数对不上,抛出形状不匹配的 RuntimeError。

## 修复方案
改 `vllm/model_executor/models/midashenglm.py` 的 `DashengAttention.forward`:
1. QKV reshape:`C // self.num_heads` 改为 `self.head_dim`(TP 分片后的每头维度)。
2. 输出 reshape:`reshape(B, N, C)` 改为 `reshape(B, N, self.q_size)`(TP 分片后的 q 尺寸)。
3. 顺带把未使用的 `C` 改名为 `_`。

关键 diff 摘录:
```diff
-        B, N, C = x.shape
+        B, N, _ = x.shape
         qkv, _ = self.qkv(x)
-        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
+        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
...
-        x = x.transpose(1, 2).reshape(B, N, C)
+        x = x.transpose(1, 2).reshape(B, N, self.q_size)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响所有 TP>1 部署 midashenglm-7b 的场景。
- 回归面:仅 `vllm/model_executor/models/midashenglm.py` 一个 forward 函数的三行;TP=1 时 `self.head_dim == C // num_heads`、`self.q_size == C`,行为不变(正文确认 TP=1 前后均正常)。TP>1 时从崩溃变为正常。风险集中在该类分片属性(`head_dim`/`q_size`)与投影层实际分片方式的一致性上。

## 关联
正文未列出 issue/PR 链接(附了 serve_succeed.log / serve_failed.log 两个测试日志附件)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
