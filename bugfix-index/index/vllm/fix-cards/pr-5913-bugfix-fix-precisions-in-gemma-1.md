# PR #5913: [Bugfix] Fix precisions in Gemma 1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5913](https://github.com/vllm-project/vllm/pull/5913) |
| 作者 | WoosukKwon |
| 合并日期 | 2024-06-29 |
| labels | (none) |
| 热度 | 👍 2 · 💬 8 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

Gemma 1 用了精度不足的 RMSNorm 与通用 RoPE 实现;改为专用的 GemmaRMSNorm / GemmaRotaryEmbedding 并移除 +1.0 权重 hack,以速度换精度。

## 问题现象

正文未描述具体现象,仅说明本 PR 修复 Gemma 1 的精度问题,并明确提示:由于 Gemma 1 改用未优化(但更准确)的 `GemmaRMSNorm`,合并后 Gemma 1 的速度可能下降。PR 需在 #5908 之后合并。

## 根因

正文未系统披露根因,仅在 NOTE 中说明此前实现精度不足、本 PR 改用更准确的 `GemmaRMSNorm`。(由 diff 推断)具体机制:vLLM 的 `gemma.py` 此前对 Gemma 1 复用了普通 `RMSNorm`,并在 `load_weights` 里以 `loaded_weight += 1.0` 的 hack 弥补差异——被删除的原注释写明「GemmaRMSNorm is different from Llama's in that it multiplies (1 + weight) to the output, instead of just weight」,即权重加一的处理发生在加载期而非算子内,精度处理不够忠实;同时 rotary embedding 走通用 `get_rope` 接口而非 Gemma 专用实现。

## 修复方案

改 `vllm/model_executor/models/gemma.py`:

- `input_layernorm`、`post_attention_layernorm`、最终 `norm` 全部由 `RMSNorm` 换成 `GemmaRMSNorm`;
- rotary embedding 从 `get_rope(...)` 换成直接构造 `GemmaRotaryEmbedding(head_dim, rotary_dim=head_dim, max_position_embeddings=..., base=rope_theta, is_neox_style=True, dtype=torch.get_default_dtype())`,并留 TODO(woosuk): Use the `get_rope` interface;
- `load_weights` 删除 `if "norm.weight" in name: loaded_weight += 1.0` 的补丁(改用 GemmaRMSNorm 后无需再加 1);
- `GemmaForCausalLM.forward` 移除 `@torch.no_grad()` 装饰器。

`tests/models/test_models.py` 的模型测试清单中加入 `google/gemma-1.1-2b-it`。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:所有 Gemma 1 模型的前向路径。正文明示速度可能下降(RMSNorm→未优化的 GemmaRMSNorm);数值结果会与旧实现不同(+1.0 逻辑从加载期移入算子、RoPE 实现更换、显式传入 dtype);移除 `@torch.no_grad()` 改变 forward 的梯度上下文;`get_rope` 换为直接构造绕过了通用选择逻辑(正文留 TODO 未迁移),若其他 Gemma 变体共享此文件路径需确认不受影响。

## 关联

- #5908(需先合并的前置 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
