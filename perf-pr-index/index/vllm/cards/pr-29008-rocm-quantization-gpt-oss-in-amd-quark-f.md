# PR #29008: [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations 

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xuebwang |
| 合并 | 2026-02-10 |
| 分类 | Quantization \| tier: major |
| Label | rocm, ready, gpt-oss |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 3 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/29008 |

## 一句话结论

为 GPT_OSS 增加 amd-quark 格式加载与 OCP-MX MoE 量化/仿真，统一 mxfp4 与 quark 权重加载路径。

## 动机

PR 正文 Purpose 说明，此前 vLLM 仅对 openai/gpt-oss-20b 与 openai/gpt-oss-120b 提供 `mxfp4` 格式加载（gpt_oss.py 中的加载函数），缺少 amd-quark 格式的端到端支持。本 PR 目标是把 quark 模型加载与既有 `mxfp4` 加载结合，并把 OCPMX_W4A16、OCPMX_W4AFP8 两种 MoE 量化 scheme 及 emulation forward 统一进 `QuarkOCP_MX_MoEMethod`。子任务还列出统一 `mxfp4` 与 `quark` 格式加载（layer.py 的 FIXME）、修复 issue #30621 引入的衍生问题、重构 `quark_moe.py` 以兼容 W4A16 与 FP8 方法。

## 优化点

本 PR 属于量化与模型加载功能扩展，PR 正文未给出性能优化机制或数字。据 PR 正文内的 Bugbot 摘要，机制改动为：统一 GPT-OSS MoE 权重加载，覆盖 fused experts、per-expert mapping、KV-cache scales、bias 与 EP/TP-aware slicing；新增 `mxfp4_w4a16_moe_quant_config`、`mxfp4_fp8_moe_quant_config`，并让 fp8/int8 w8a8 config 接受 bias；扩展 OCP-MX schemes（含 `w_mxfp4` 与 `*_a_fp8`）；泛化 fused MoE runtime，增加 input quant dtype 检测、MXFP4/MXFP6 dequant 与 FP8 activation QDQ emulation；GPT-OSS Triton MoE 路径接入 `PrecisionConfig`。

## 关键改动

```text
vllm/model_executor/layers/quantization/quark/quark_moe.py: 重构出 QuarkOCP_MX_MoEMethod，支持 W4A16/W4Afp8、bias 与静态 FP8 input scale，按 backend 路由到 ROCm AIter/Triton/native
vllm/model_executor/layers/fused_moe/config.py: 新增 mxfp4_w4a16/mxfp4_fp8_moe_quant_config，并让 fp8_w8a8/int8_w8a8 quant config 接受 bias
vllm/model_executor/layers/fused_moe/fused_moe.py: 泛化运行时，增加 input quant dtype 检测、MXFP4/MXFP6 dequant 与 FP8 activation QDQ emulation
vllm/model_executor/layers/fused_moe/layer.py: hidden-size roundup 限定 gpt_oss + MXFP4，扩展 GPT-OSS fused/bias 权重加载
tests/models/quantization/test_gpt_oss.py: 新增 GPT-OSS 量化模型 E2E GSM8k 精度测试（diff 中新建）
```

```python
if a_dtype == "bf16" and w_dtype == "mx4":
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=pc1,
        w2_scale=pc2,
        w1_bias=w1_bias_tri,
        w2_bias=w2_bias_tri,
    )
```

（摘自 `tests/kernels/moe/test_gpt_oss_triton_kernels.py`：等价性测试改用新的 quant config 构造器。）

## 性能声明

PR 正文未给出定量性能数据。正文 Test Result 仅以图片链接（user-attachments 的 img，宽 1418、高 304）呈现，pack 的 perf claims regex 也未检测到数字。Test Plan 覆盖 GPT_OSS_20B/GPT_OSS_120B、W:MXFP4 搭配 A:BF16/FP8/MXFP4（可选 KV:FP8）以及 TP1/TP2/TP4/TP8。

## 验证与回归风险

PR 引入/依赖的测试：新增 GPT-OSS MoE MXFP4 的 Triton kernel 等价性测试（`test_gpt_oss_triton_kernels.py` 改用 `mxfp4_w4a16_moe_quant_config`）与 GPT-OSS 量化模型 E2E GSM8k 精度测试（`test_gpt_oss.py`），并移除旧的 attention-only 测试；Test Plan 列出上述模型、scheme 与 TP 组合。PR 未注明回归风险；改动集中在 GPT-OSS MoE 权重加载、fused MoE runtime 与 quark/mxfp4 quant config，可能受影响的路径为 GPT-OSS 量化模型加载与 forward、ROCm AIter/Triton/native backend 选择。

## 关联

正文提及 issue #30621（衍生问题，已在 commit c4a05ed 修复）与 PR #31676（gpt_oss 的 W-FP8-AFP8 forward 修复），并明确 W-MXFP4-A-MXFP8 scheme 不在本 PR 范围。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
