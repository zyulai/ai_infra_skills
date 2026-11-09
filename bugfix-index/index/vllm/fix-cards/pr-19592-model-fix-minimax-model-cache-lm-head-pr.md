# PR #19592: [Model] Fix minimax model cache & lm_head precision

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19592](https://github.com/vllm-project/vllm/pull/19592) |
| 作者 | qscqesze |
| 合并日期 | 2025-06-13 |
| labels | ready |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
MiniMax 的 lm_head 与 KV cache 停在 bf16 精度不足;cache、lm_head 与 logits 输入统一转 fp32 提升数值稳定性。

## 问题现象
正文未描述具体现象,仅说明目的:把 vLLM 中 MiniMax 模型的 LM head 与 KV cache 从 bfloat16 更新为 float32,以改善推理的数值稳定性与输出精度。

## 根因
正文未披露根因。(由 diff 推断)`MinimaxCacheManager` 以模型自身 dtype(即 bf16)管理 cache,`compute_logits` 直接把 bf16 的 `hidden_states` 交给 logits_processor,`lm_head` 权重亦保持 bf16,相关计算的数值精度受限。

## 修复方案
`vllm/model_executor/models/minimax_text_01.py` 三处改动:
1. `MinimaxCacheManager(dtype=self._dtype, ...)` 改为 `dtype=torch.float32`;
2. 模型构造中新增 `self.lm_head.float()`;
3. `compute_logits` 先把 hidden_states 转 fp32 再计算 logits:

```diff
-        self.minimax_cache = MinimaxCacheManager(dtype=self._dtype,
+        self.minimax_cache = MinimaxCacheManager(dtype=torch.float32,
...
-        logits = self.logits_processor(self.lm_head, hidden_states,
+        logits = self.logits_processor(self.lm_head, hidden_states.float(),
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `minimax_text_01.py` 模型路径:KV cache 存储 dtype 从模型 dtype 变为固定 fp32,logits 计算的输入从 bf16 转 fp32,精度提升的同时 cache 与中间量的内存占用相应增大(由 diff 推断);`lm_head` 权重被就地转 fp32。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
