# PR #29757: Add Mistral Large 3 and Ministral 3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @juliendenize |
| 合并 | 2025-12-02 |
| 分类 | Model-specific Fast Path(major) |
| Label | documentation, new-model, frontend, speculative-decoding, ready, v1, tool-calling, deepseek |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 1 个超大文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/29757 |

## 一句话结论

新增 Mistral Large 3(675B MoE)与 Ministral 3 支持:复用 DeepSeek-V3 实现路径、EAGLE 投机解码 draft 模型,并附 H200 FP8 fused MoE 调优配置。

## 动机

Mistral 发布 Mistral-Large-3-675B(Base/Instruct,MoE 带 shared experts 与 llama_4_scaling)与 Ministral-3-3B(Mistral 格式)。两者使用 Mistral 自有的权重命名与配置结构,若不接入则无法在 vLLM 上运行;将 Mistral 命名映射到已优化的 DeepSeek-V3 代码路径,可直接继承其 MLA、fused MoE、投机解码等基础设施与性能。

## 优化点

`MistralLarge3ForCausalLM` 继承 `DeepseekV3ForCausalLM`,加载权重时用正则表把 Mistral 命名(wq_a/wkv_b/experts.*/shared_experts 等)映射到 DeepSeek 命名,并把 `.qscale_act`/`.qscale_weight` 改写为 `input_scale`/`weight_scale`。config 侧:mistral config 中 `moe.num_shared_experts > 0` 时判定为 Mistral Large 3,`_remap_moe_args` 把 MoE 字段映射为 DeepSeek 字段(n_routed_experts、moe_intermediate_size 等),model_type 改为 `deepseek_v3`,并强制要求 `llama_4_scaling`。`deepseek_v2.py` 新增 llama_4_scaling:`q *= 1 + beta*log(1+floor(pos/original_max_position))`,每层 forward 前计算一次逐层传递;RoPE 支持 `deepseek_llama_scaling` 类型(不施加 yarn mscale)。另支持 EAGLE draft 模型(`EagleMistralLarge3ForCausalLM`,fc 融合 embedding 与 hidden states),FP8 量化 remap 支持 `qscheme_act` 为 NO_SCALES(dynamic)/TENSOR(static);新增 fused MoE 调优配置 `E=128,N=512,NVIDIA_H200,fp8_w8a8`。Ministral 3 走既有 `MistralForCausalLM`(Llama 架构)与 Pixtral 多模态路径,并修 mistral tool parser 正则与 tokenizer 字段过滤。

## 关键改动

```text
vllm/model_executor/models/mistral_large_3.py: Mistral→DeepSeek 权重名映射与 FP8 scale 名改写
vllm/model_executor/models/mistral_large_3_eagle.py: EAGLE draft 模型(fused fc + DeepseekV2DecoderLayer)
vllm/transformers_utils/configs/mistral.py: MoE config 识别/映射与 llama_4_scaling 校验
vllm/model_executor/models/deepseek_v2.py: llama_4_scaling 计算与逐层传递;deepseek_llama_scaling rope 类型
vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json: H200 tuned MoE 配置
```

llama_4_scaling 核心:

```python
def _get_llama_4_scaling(original_max_position_embeddings, scaling_beta, positions):
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings))
    return scaling[..., None, None]
```

## 性能声明

PR 正文未提供任何 benchmark 数字(无性能声明);H200 fused MoE 配置为调优产物,但未附测得加速比。测试条件无法引用。

## 验证与回归风险

`tests/models/registry.py` 注册 `MistralLarge3ForCausalLM`、`EagleMistralLarge3ForCausalLM` 与 Pixtral extras(因模型未公开均 `is_available_online=False`);`tests/tokenizers_/test_mistral.py` 新增工具字段过滤用例。风险面:改动共享的 `deepseek_v2.py` 与 mistral config 适配逻辑——`apply_yarn_scaling` 默认 True 时保持原 `deepseek_yarn` 行为,llama_4_scaling 默认 None 不触发,但 DeepSeek 系模型加载路径受 config remap 重构(`_remap_mistral_quantization_args`)影响。

## 关联

同区域:DeepSeek-V3/MLA/MoE fast path 与 EAGLE 投机解码系列;fused MoE 调优配置类 PR(如 #34846)。具体前序/后续编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
