# PR #41574: [Model] Fix Gemma4 MoE activation mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41574](https://github.com/vllm-project/vllm/pull/41574) |
| 作者 | lucianommartins |
| 合并日期 | 2026-05-05 |
| labels | ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
`Gemma4MoE` 传了 `activation="gelu"`(精确 GELU),而模型规格与 HF 参考实现均为 tanh 近似变体;为 `MoEActivation` 补上 `GELU_TANH` 并改用之。

## 问题现象
Gemma4 的 MoE 路径与官方模型规格不一致:`Gemma4MoE` 给 `FusedMoE` 传 `activation="gelu"`(经 `torch.ops._C.gelu_and_mul` 的精确 GELU),但发布的模型 config(`hidden_activation="gelu_pytorch_tanh"`)与 HuggingFace 参考实现用的都是 tanh 近似变体;同一文件里的稠密 `Gemma4MLP` 已用 `raise ValueError` 守卫强制 `gelu_pytorch_tanh`。影响量级:精确与 tanh 近似 GELU 在 float32 下每元素差约 2e-4,在常规生成的 bf16 噪声之内——主要是实现与规格不符的正确性问题,非性能问题(两 kernel 速度相同)。正文未描述运行时报错。

## 根因
正文披露:编写 `Gemma4MoE` 时 `MoEActivation` 枚举还没有 `GELU_TANH` 变体,于是用了当时最接近的字符串 `"gelu"`。

## 修复方案
- `vllm/model_executor/layers/fused_moe/activation.py`:枚举新增 `GELU_TANH` 与 `GELU_TANH_NO_MUL`;`from_str()` 接入 `_STR_ALIASES`(`"gelu_pytorch_tanh" → "gelu_tanh"`);gated 分派到 `torch.ops._C.gelu_tanh_and_mul`,非 gated 到 `F.gelu(input, approximate="tanh")`;注册进 `_CUSTOM_OP_NAMES` 与 `_WITHOUT_MUL` 查找表。
- 四个后端的 `_supports_activation` 白名单(TritonExperts、BatchedTritonExperts、MarlinExpertsBase、HummingExpertsBase)加入两个新枚举。
- `vllm/model_executor/models/gemma4.py`:`Gemma4MoE.__init__` 的 `activation="gelu"` 改为 `activation="gelu_tanh"`。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:FusedMoE 激活枚举与四个 MoE 后端白名单属共享代码,但只新增可选激活、不改既有枚举行为;唯一改变数值行为的是 Gemma4 MoE 层(gelu → gelu_tanh,使输出与 HF 参考一致)。验证:ruff 检查通过、`tests/kernels/moe/test_gemma4router.py` 12/12 通过、冒烟测试确认 `from_str("gelu_tanh")` 与 `from_str("gelu_pytorch_tanh")` 均解析到 `GELU_TANH`,输出匹配 `F.gelu(approximate="tanh")` 且与 `GELU` 输出如预期发散。

## 关联
- [google/gemma-4-27B-it config.json](https://huggingface.co/google/gemma-4-27B-it/blob/main/config.json)(规格依据,`hidden_activation: gelu_pytorch_tanh`)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
