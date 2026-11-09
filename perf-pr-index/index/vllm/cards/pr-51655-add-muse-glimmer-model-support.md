# PR #51655: Add Muse Glimmer model support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xianbaoqian |
| 合并 | 2026-08-14 |
| 分类 | Speculative Decoding |
| Label | documentation, new-model, speculative-decoding, ready, tool-calling, qwen, mrv2 |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 17 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/51655 |

## 一句话结论

新增 Muse Glimmer 29.6B 视觉语言模型支持,并为其 draft head 接入 DFlash speculative decoding(每次 forward 预测 16-slot 块)。

## 动机

Muse Glimmer 是一个 dense 29.6B 视觉语言模型(ViT-G/14 感知编码器,128K context),此前 vLLM 无法运行。其输出格式特殊:不产出 JSON tool call、不用 `<think>` 包裹推理,每轮是 channel-scoped 消息序列,需要配套 parser;同时官方 draft head 走 DFlash 投机解码,需要接入 vLLM 的 spec-decode 配置与接口。

## 优化点

主体是模型接入而非内核优化:新增 `MuseGlimmerForConditionalGeneration`/`MuseGlimmerForCausalLM` 原生实现(文本栈为 Gemma2 衍生,含 SiLU-gated MLP、带 `+1` weight offset 的 sandwich RMSNorm、RoPE 前的 fp32 QK-norm 与 query pre-scale、per-head sigmoid attention output gate、iRoPE(NoPE 层全注意力 + RoPE 层 sliding window)、logit tanh soft-cap、untied lm_head;视觉路径支持变分辨率图像与时间分片视频)。性能相关部分:DFlash draft head(`MuseGlimmerAssistantModel`)复用现有 qwen3_dflash 实现,读取 target 在 layers [1, 13, 25, 37, 49] 的 residual stream,每次 forward 预测一个 16-slot 块;`vllm/config/speculative.py` 的 method 自动探测新增按 architectures 含 `MuseGlimmerAssistantModel` 判定为 `dflash`;`interfaces.py` 修正 multimodal 模型 set/get aux hidden state layers 时对 `EagleModelMixin` 的解包逻辑(仅在有 `.model` 可解包时解包)。

## 关键改动

```text
vllm/model_executor/models/muse_glimmer.py: 新增 1649 行模型实现(文本栈 + 视觉编码器 + 多模态 processor)
vllm/config/speculative.py: draft 模型 architectures 含 MuseGlimmerAssistantModel 时自动选 dflash
vllm/model_executor/models/interfaces.py: 修复 multimodal 模型 aux hidden state layers 的解包
tests/models/registry.py: 注册 4 个 arch 的测试条目(speculative_model=Muse-Glimmer-30B-assistant)
```

channel-scoped reasoning 与 ATEM tool-call parser、processor 等在被裁剪的 17 个文件内。

## 性能声明

PR 正文未提供任何 benchmark 数字或加速比,未注明测试条件。DFlash 投机解码的接受率/加速效果 PR 未注明。

## 验证与回归风险

依赖 tests/models/registry 注册的模型测试(`MuseGlimmerForConditionalGeneration`、`MuseGlimmerForCausalLM`、`MuseGlimmerAssistantModel`、`DFlashMuseGlimmerAssistantModel`,CI 低显存环境将 max_model_len 降到 8192、max_num_seqs 32)。风险:`interfaces.py` 的 aux hidden state 解包改动影响所有走 `EagleModelMixin` 的 multimodal 模型路径;reasoning parser 强制 `skip_special_tokens=False`,否则 marker 被剥离导致两个 channel 塌缩进 content;`tool_choice="required"` 与命名 tool_choice 通过 `supports_required_and_named=False` 关闭 JSON guided decoding,避免调用被困在 reasoning channel。

## 关联

无(复用 qwen3_dflash 投机解码实现;本批索引中无同区域前后序 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
