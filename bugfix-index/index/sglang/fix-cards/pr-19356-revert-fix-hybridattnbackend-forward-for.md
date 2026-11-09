# PR #19356: Revert "Fix HybridAttnBackend forward for linear attention"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19356](https://github.com/sgl-project/sglang/pull/19356) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2026-02-25 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
PR #19006 给 HybridAttnBackend 加的统一 forward 引发 CI 失败,本 PR 整体回滚该方法。

## 问题现象
正文披露:被回滚的 PR #19006 导致 CI 失败,失败日志见链接(GitHub Actions run 22411646175 的 job 64887051730,关联 PR #14105)。正文未给出具体报错文本与失败测试名。

## 根因
正文只指出 #19006 引发 CI 失败,未披露具体失败机制。

## 修复方案
整体回滚 PR #19006 的代码改动:删除 `python/sglang/srt/layers/attention/hybrid_attn_backend.py` 中 `HybridAttnBackend` 新增的 `forward` 方法(位于 `get_indexer_metadata` 之后)。该方法原先同时支持常规 attention(q/k/v)与 linear attention(`mixed_qkv`/`a`/`b`),内部按 `forward_batch.forward_mode` 用 `_select_backend` 选中后端并把全部参数透传过去:
```diff
-    def forward(
-        self,
-        q: Optional[torch.Tensor] = None,  # For full attention
-        k: Optional[torch.Tensor] = None,  # For full attention
-        v: Optional[torch.Tensor] = None,  # For full attention
-        layer: RadixAttention = None,
-        forward_batch: ForwardBatch = None,
-        save_kv_cache: bool = True,
-        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
-        a: Optional[torch.Tensor] = None,  # For linear attention
-        b: Optional[torch.Tensor] = None,  # For linear attention
-        **kwargs,
-    ):
-        """Forward method that supports both regular attention (q, k, v) and linear attention (mixed_qkv, a, b)."""
-        backend = self._select_backend(forward_batch.forward_mode)
-        return backend.forward(...)
```
回滚后 `HybridAttnBackend` 恢复到 #19006 之前的状态,linear attention 的调用路径不再经过这个统一 forward。

## 影响与回归风险
- 影响窗口:#19006 合并之后至本 PR 回滚之间。
- 回归面:混合注意力后端(`hybrid_attn_backend.py`)的 forward 分发路径恢复原状;此前依赖 #19006 新增 forward 的 linear attention 模型调用方会随之回到旧调用方式——若 #19006 期间有其他代码已开始使用该接口,回滚可能使其失效(正文未说明是否有此类依赖)。

## 关联
- https://github.com/sgl-project/sglang/pull/19006(被回滚的 PR)
- https://github.com/sgl-project/sglang/actions/runs/22411646175/job/64887051730?pr=14105(CI 失败日志)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
