# PR #34110: [MODEL] Adding Support for Qwen3.5 Models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JJJYmmm |
| 合并 | 2026-02-09 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, new-model, speculative-decoding, ready, v1, qwen |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含新增 qwen3_5.py(约 993 行)与 5 个文件修改,另有 6 个文件未展示 |
| 链接 | https://github.com/vllm-project/vllm/pull/34110 |

## 一句话结论

为 vLLM 接入 Qwen3.5 系列(dense 与 MoE、hybrid GDN+full attention、多模态),并打通其 MTP 投机解码支持;PR 未附 benchmark 数字。

## 动机

Qwen3.5 即将发布,需要 vLLM 侧的推理支持:该系列是 hybrid 架构(linear_attention 层走 GDN/Gated DeltaNet,full_attention 层走标准 attention),dense 与 MoE 两种变体,外加视觉塔的多模态形态,并自带 MTP(Multi-Token Prediction)模块。HF 参考实现在 transformers#43830。

## 优化点

新增 vllm/model_executor/models/qwen3_5.py(约 993 行):Qwen3_5GatedDeltaNet 复用 Qwen3NextGatedDeltaNet 基建(conv1d/in_proj_qkvz/in_proj_ba、RMSNormGated、gdn_attention_core custom op),但明确不需要 fix_query_key_value_ordering;Qwen3_5DecoderLayer 按 layer_types 在 linear_attention(GDN)与 full_attention 间选择,MLP 按 model_type 走 dense Qwen3NextMLP 或 Qwen3NextSparseMoeBlock;Qwen3_5ForConditionalGeneration 复用 Qwen3VL 的视觉塔与多模态 processor。投机解码:speculative.py 注册 qwen3_5_mtp,并在 hf_config_override 中把 model_type qwen3_5/qwen3_5_moe 映射为 Qwen3_5MTP/Qwen3_5MoeMTP(n_predict 取 mtp_num_hidden_layers);mamba/abstract.py 为这三个 model_type 放行 Mamba+speculative 组合。另修正 config/model.py 的 hybrid 层计数(layer_types 改从 hf_text_config 读取)与 fused expert 权重加载路径。

## 关键改动

```text
vllm/model_executor/models/qwen3_5.py: 新增,Qwen3.5 dense/MoE/多模态模型实现(GDN + Qwen3Next 复用)
vllm/config/speculative.py: qwen3_5_mtp 注册与 hf_config_override 的 MTP 架构映射
vllm/model_executor/layers/mamba/abstract.py: qwen3_5 系允许 Mamba + speculative decoding
vllm/config/model.py: hybrid 层计数改读 hf_text_config.layer_types
docs/models/supported_models.md + tests/models/registry.py: 注册 Qwen3.5-9B/35B-A3B 及 MTP 测试条目
```

## 性能声明

PR 正文未提供任何 benchmark 或性能数字(正文仅为支持声明与致谢)。性能相关点体现在:GDN 层走 vLLM 现有 gdn_attention_core custom op 与 torch.compile 支持(support_torch_compile 装饰),MTP 投机解码可用;代码中注明 Qwen3.5 不支持 mamba-cache-mode=all 的 prefix caching,需用 align 模式。

## 验证与回归风险

tests/models/registry.py 新增四个条目(Qwen3_5ForConditionalGeneration、Qwen3_5MoeForConditionalGeneration 及两个 MTP,要求 transformers>=5.1.0);6 个文件未在 pack diff 中展示,测试细节不可见。回归面:hybrid 层计数逻辑改动影响所有使用 layer_types 的 hybrid 模型(Qwen3Next 等);Mamba+spec 放行只对新 model_type 生效;fused expert 加载的 chunk(dim=-2)路径对权重布局敏感。

## 关联

同区域/后续:PR #42311(Qwen3.5 GDN 输出投影 einops->flatten 优化,2026-05-18 合并);PR #34130(fused_moe int4_w4a16 benchmark/tuning,同批次 MoE 方向)。前序:Qwen3-Next 系列(代码复用其 GDN/attention/MoE 组件)。
