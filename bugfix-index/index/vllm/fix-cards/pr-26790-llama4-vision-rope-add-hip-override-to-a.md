# PR #26790: llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26790](https://github.com/vllm-project/vllm/pull/26790) |
| 作者 | hl475 |
| 合并日期 | 2025-10-14 |
| labels | bug, llama, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

Llama4-Vision RoPE 只实现了 2 参 `forward_cuda`,HIP 调 3 参基类路径导致实参错位;补 `forward_hip(q, k)` 修复。

## 问题现象

ROCm/HIP 上使用 Llama4 视觉模型时,RoPE 调用参数不匹配。基类 RoPE 路径(`rotary_embedding/base.py`)期望 `(positions, query, key)` 三参,并在 fallback 分支调用 `forward_cuda(positions, query, key)`;而 `Llama4VisionRotaryEmbedding` 实现的 `forward_cuda(query, key)` 只有 2 个参数。因此在 HIP 上可能以错误的实参顺序或错误的参数个数调用,正文未给出具体报错日志。

## 根因

PR 正文披露:`Llama4VisionRotaryEmbedding` 覆写了签名不同的 `forward_cuda(query, key)`(2 参,`type: ignore[override]`),与基类 `forward_cuda(positions, query, key)`(3 参)不一致;HIP 分派走基类逻辑时会把 `positions` 与 `query`/`key` 错位传入,或传参数量不对,造成实参不匹配。

## 修复方案

在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` 中为 `Llama4VisionRotaryEmbedding` 新增 `forward_hip(query, key)` 覆写,直接委托给已有的 `forward_native` 实现:

```diff
         return self.forward_native(query, key)
+
+    def forward_hip(  # type: ignore[override]
+        self,
+        query: torch.Tensor,
+        key: torch.Tensor | None = None,
+    ) -> tuple[torch.Tensor, torch.Tensor | None]:
+        return self.forward_native(query, key)
```

这样保持调用点 `vllm/model_executor/models/mllama4.py` 的 `self.rotary_emb(q, k)` 不变,避免构造 `positions`,并阻止 HIP 走基类 3 参 CUDA 路径。

## 影响与回归风险

- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 ROCm/HIP 上的 Llama4 Vision RoPE 分派路径(新增一个 override);CUDA 路径的 `forward_cuda` 与 native 实现未改动,其他模型的 RoPE 不受影响。

## 关联

正文未列出(正文只引用了相关代码行链接,未列 issue/PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
