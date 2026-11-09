# PR #37550: [Bugfix] Fix CPU backend crash in KV cache block zeroing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#37550](https://github.com/vllm-project/vllm/pull/37550) |
| 作者 | DorBernsohn |
| 合并日期 | 2026-03-23 |
| labels | bug, cpu, ready, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
CPU 后端首个推理请求即崩:Triton 清零 kernel 无 CPU 回退;CPUModelRunner 覆盖 `_zero_block_ids` 修复。

## 问题现象
CPU 后端上所有模型在处理第一个推理请求时崩溃,报 `TypeError: 'function' object is not subscriptable`。正文未描述更多环境细节,仅说明影响 CPU backend 的全部模型。Closes #37546。

## 根因
正文披露:Triton block-zeroing kernel(`_zero_kv_blocks_kernel`)由 #35219(3 月 10 日)引入,但 `CPUModelRunner` 没有提供 CPU 安全的回退实现,CPU 节点上(无活动 GPU 驱动)调用该 Triton GPU kernel 直接崩溃。

## 修复方案
在 `vllm/v1/worker/cpu_model_runner.py` 中覆盖 `_zero_block_ids`,使其不再走到基类的 Triton kernel 路径。注意:正文称用 PyTorch(`tensor.zero_()`)实现覆盖,但实际合并 diff 中是空实现(`pass`),并附注释说明理由(照抄):「CPU attention assigns -INF to logits at invalid positions, so stale KV cache data never affects computation.」

关键 diff 摘录:
```diff
+    def _zero_block_ids(self, block_ids: list[int]) -> None:
+        # CPU attention assigns -INF to logits at invalid positions,
+        # so stale KV cache data never affects computation.
+        pass
```

## 影响与回归风险
- 影响窗口:#35219 合并(3 月 10 日)之后。
- 回归面:仅 CPU 后端的 KV cache 块清零路径被改为 no-op;按 diff 注释,CPU attention 对无效位置 logits 赋 -INF,陈旧 KV 数据不影响计算结果,但若该前提在某个 CPU attention 实现中不成立,可能引入精度问题。

## 关联
- Closes [#37546](https://github.com/vllm-project/vllm/issues/37546)
- 引入问题的 [#35219](https://github.com/vllm-project/vllm/pull/35219)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
