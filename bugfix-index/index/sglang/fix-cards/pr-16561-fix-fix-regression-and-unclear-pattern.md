# PR #16561: fix: fix regression and unclear pattern

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16561](https://github.com/sgl-project/sglang/pull/16561) |
| 作者 | nanjiangwill |
| 合并日期 | 2026-01-17 |
| labels | high priority, run-ci |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
`post_residual_addition` 经 `**kwargs` 隐式透传,部分路径丢失该参数导致回归;改为全链路显式传参修复。

## 问题现象
正文未描述具体现象。Motivation 仅说明:修复 PR #14636 评审中的一条 comment(#discussion_r2660730120),对应标题里的两件事——「fix regression」(近期改动引入的回归)与「unclear pattern」(指 `**kwargs` 隐式传参的不清晰模式)。Modifications 一节留空,未给出复现条件或日志。

## 根因
正文未披露根因,仅链接到 PR #14636 的 review comment。(由 diff 推断) 此前 `post_residual_addition`(残差相加后再叠加的张量)沿 communicator → layernorm → 各后端 forward 的调用链用 `**kwargs` 隐式传递,存在两类问题:一是 `forward_with_allreduce_fusion` 的兜底路径调用 `self.forward(x, residual)` 时完全丢弃了 `post_residual_addition`,使该参数在 allreduce fusion 路径下失效,残差计算与预期不一致;二是 `forward_native` 等内部再用 `kwargs.get("post_residual_addition")` 取值,签名不透明,中间任何一层忘记透传都会静默丢参,这正是 regression 的来源。

## 修复方案
把 `post_residual_addition` 从 `**kwargs` 改为整条链路上的显式可选参数:

- `python/sglang/srt/layers/communicator.py`:`prepare_attn_and_capture_last_layer_outputs` 与 `prepare_attn` 显式声明并转发该参数;`input_layernorm(hidden_states, **kwargs)` 改为 `input_layernorm(hidden_states)`,带残差分支改为 `input_layernorm(hidden_states, residual, post_residual_addition)`。
- `python/sglang/srt/layers/layernorm.py`:RMSNorm 的 `forward_cuda/forward_npu/forward_aiter/forward_hip/forward_native/forward_cpu/forward_xpu/forward_with_allreduce_fusion` 以及 LayerNorm、GemmaRMSNorm、Gemma3RMSNorm 全部改为显式签名,删除 `kwargs.get(...)` 写法;`forward_native` 中 `x + residual + post_residual_addition` 的混合表达式改写为先加 residual 再按需加 `post_residual_addition`。
- `python/sglang/srt/models/qwen3.py`:`Qwen3DecoderLayer.forward` 显式接收 `post_residual_addition` 并传给 `prepare_attn`。
- `python/sglang/srt/layers/quantization/modelslim/modelslim.py`:`_rmsnorm_forward_oot` 签名同步补上该参数。

关键 diff 摘录(layernorm.py):

```diff
-        return self.forward(x, residual)
+        return self.forward(x, residual, post_residual_addition)
```

```diff
-            post_residual_addition = kwargs.get("post_residual_addition")
             if post_residual_addition is not None:
                 residual = residual + post_residual_addition
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;从 review comment 看,回归由 PR #14636 引入。
- 回归面:改动集中在 `layers/communicator.py`、`layers/layernorm.py`、`models/qwen3.py`、`layers/quantization/modelslim/modelslim.py` 的函数签名与传参,覆盖 CUDA/HIP(Aiter)/NPU/CPU/XPU 多后端的 RMSNorm、LayerNorm、GemmaRMSNorm、Gemma3RMSNorm 实现。所有经 `prepare_attn` 进入 layernorm 的前向路径行为受影响,尤其是带 `post_residual_addition` 的路径(allreduce fusion 兜底、`rl_on_policy_target == "fsdp"` 的 batch-invariant 分支等);若仍有调用方以旧 `**kwargs` 方式传参,将直接抛 TypeError 而非静默丢参。

## 关联
- PR #14636 review comment: https://github.com/sgl-project/sglang/pull/14636#discussion_r2660730120

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
