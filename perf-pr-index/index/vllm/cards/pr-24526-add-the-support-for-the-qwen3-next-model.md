# PR #24526: Add the support for the qwen3 next model (a hybrid attention model).

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sighingnow |
| 合并 | 2025-09-11 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, new-model, speculative-decoding, ready, v1, qwen |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 23 个文件(16 个超长文件被省略),新增 qwen3_next.py 约 1294 行 |
| 链接 | https://github.com/vllm-project/vllm/pull/24526 |

## 一句话结论

为 vLLM 引入 Qwen3-Next(hybrid attention)模型支持,含 GDN 线性注意力路径与 MTP 投机解码接入。

## 动机

PR 正文为空(素材中未包含正文),动机无法从正文引用;从标题与 diff 可确认目的:让 `Qwen/Qwen3-Next-80B-A3B-Instruct` 等 Qwen3NextForCausalLM 模型可在 vLLM 上推理。该模型是 full_attention 与 linear_attention(gated delta net)混合的 hybrid 架构,需要新的 attention backend、状态管理与投机解码接线,现有机型代码不能直接复用。

## 优化点

新增 `Qwen3NextForCausalLM` 模型实现:`Qwen3NextGatedDeltaNet` 层复用 `vllm.model_executor.layers.fla.ops` 的 `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` Triton 算子做线性注意力,forward 经新注册的 `vllm.gdn_attention` custom op(加入 CompilationConfig 的 fused ops 列表,便于 torch.compile 切分),attention backend 指向 `GDNAttentionBackend`;MoE 部分复用 FusedMoE + Qwen2MoeMLP。状态管理新增 `gated_delta_net_state_dtype` / `gated_delta_net_state_shape` 计算并考虑 spec decode 的 num_spec。为支持 MTP 投机解码,`causal_conv1d_update` Triton 核新增 `IS_SPEC_DECODING` 路径:按 `num_accepted_tokens` 对 conv state 做滑动窗口滚动(接受 k 个 token 时从对应偏移读取历史),并支持 conv_state_indices 的 stride 与更宽松的形状断言。配置层新增 `qwen3_next_mtp` SpeculativeMethod(走 use_eagle 路径,draft 与 target 共享模型并对齐量化配置),layer 计数支持 hf_config 的 `layer_types` 字段区分 hybrid 层类型。同时修正 FLA Triton 算子的越界掩码:以 `o_t < T` 的显式 mask 替代 `safe_exp`(移除该函数),修复 chunk 内尾部越界元素的指数计算。

## 关键改动

```text
vllm/model_executor/models/qwen3_next.py: 新模型实现(GDN 线性注意力 + Sparse MoE + MTP 预留)
vllm/model_executor/layers/mamba/ops/causal_conv1d.py: causal_conv1d_update 增加 spec decoding 的 conv state 滚动路径
vllm/model_executor/layers/fla/ops/{chunk_delta_h,chunk_o,chunk_scaled_dot_kkt,op}.py: 移除 safe_exp,显式 o_t < T 掩码修正
vllm/config/__init__.py: qwen3_next_mtp 投机解码方法、layer_types 混合层计数
vllm/model_executor/layers/mamba/mamba_utils.py: gated_delta_net 状态 dtype/shape 计算
```

## 性能声明

PR 正文未提供 benchmark 结果(正文为空,自动检测无性能数字)。MTP 仅支持单层(`num_speculative_tokens > 1` 时打 warning 提示 Qwen3Next MTP 模型只有一层)。

## 验证与回归风险

diff 中可见的测试接线:`tests/models/registry.py` 注册 `Qwen3NextForCausalLM`(在线下载 `Qwen/Qwen3-Next-80B-A3B-Instruct`,min_transformers_version 4.56.2)与 `Qwen3NextMTP` 投机解码测试条目;文档 supported_models.md 增加条目。回归风险:FLA 算子掩码修正与 `causal_conv1d_update` 改动同样作用于其他 mamba/gated delta 用户(如 Falcon-H1、Qwen3-Next 类 hybrid 模型),state_len 计算改为 `width - 1 + (seqlen - 1)`、形状断言放宽,影响所有走 continuous batching 的 conv state 更新路径;配置层 layer_types 计数逻辑影响其他 hybrid 模型的层统计。

## 关联

同区域:Qwen3 系列模型支持(Qwen3ForCausalLM、Qwen3MoeForCausalLM 已在 registry);MTP 投机解码同方法族有 deepseek_mtp、ernie_mtp、glm4_moe_mtp、mimo_mtp;线性注意力后端为 vllm/v1/attention/backends/gdn_attn.py(GDNAttentionBackend)。pack 素材未列出明确的前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
