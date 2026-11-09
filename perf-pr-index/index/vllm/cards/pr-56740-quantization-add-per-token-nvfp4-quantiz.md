# PR #56740: [quantization] Add per-token NVFP4 quantization MoE support for ReLU2

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xuantengh |
| 合并 | 2026-10-07 |
| 分类 | MoE（tier: major） |
| Label | documentation, ready, ci/build, deepseek, nvidia, quantization, DSv4, DSv4.1 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾标注 4 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/56740 |

## 一句话结论

新增 per-token NVFP4 MoE(ReLU2) 量化，2×GB200/TP2 下 GSM8K 93.93%、吞吐 2199.9 tok/s。

## 动机

PR 目标是为 vLLM 增加带 ReLU2 激活的 per-token NVFP4 量化。正文说明 FlashInfer 上游已在 flashinfer-ai/flashinfer#4618 与 #5031 中加入 per-token NVFP4 with ReLU2 的 fused kernel 支持，本 PR 把这些新增 kernel 引入 vLLM，使 vLLM 侧具备对应的 MoE 量化路径。正文未描述改动前的具体性能瓶颈或缺陷，仅陈述「引入上游 kernel」这一目的。

## 优化点

核心是让 ModelOpt NVFP4 MoE 走动态 activation scale（per-token）路径并兼容 ReLU2 激活。diff 中 `modelopt.py` 新增 QuantKey `kNvfp4DynamicToken`，`ModelOptQuantConfig` 增加 `quantization_args` 与 `moe_activation_override`，允许通过 `quantization_config.moe.activation='nvfp4_per_token'` 覆盖；`ModelOptNvFp4FusedMoEMethod.__init__` 依据 override 设置 `self.per_token_activation = activation_key == kNvfp4DynamicToken`，并把 activation_key 传给 `select_nvfp4_moe_backend`，非法 override 直接抛 ValueError。FlashInfer TRTLLM 后端同时调整权重布局：`flashinfer_fp4_moe.py` 的 `prepare_nvfp4_moe_layer_for_fi_or_cutlass` 新增 `trtllm_hidden_alignment` 参数并下传给 `align_fp4_moe_hidden_dim_for_fi`；`flashinfer_utils.py` 新增 `trtllm_nvfp4_hidden_alignment()`，把 per-token 且非 gated 变体的 hidden 对齐从 256 提高到 512——注释说明这类 kernel 在 256 对齐 K 下 tile 配置太少，启发式默认 tactic 会报 "No valid config found"（例如 Nemotron 的 2688→2816）。测试侧相应新增 per-token 激活用例与线性化权重预处理。

## 关键改动

```text
vllm/model_executor/layers/quantization/modelopt.py: 新增 kNvfp4DynamicToken 与 moe_activation_override，设置 per_token_activation 并选择后端
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: 新增 trtllm_nvfp4_hidden_alignment()，per-token 非 gated 变体 hidden 对齐 512
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: prepare_nvfp4_moe_layer_for_fi_or_cutlass 接收并下传 hidden 对齐参数
tests/kernels/moe/test_trtllm_nvfp4_moe.py: 增加 silu_per_token、relu2_no_mul_per_token 用例与 swizzled→linear 权重转换
tests/evals/gsm8k/configs/models-blackwell-ep.txt: 注册 Nemotron-3-Super-120B-A12B-NVFP4-per-token.yaml 端到端配置

# 机制摘录 (flashinfer_utils.py)
def trtllm_nvfp4_hidden_alignment(
    per_token_activation: bool, is_act_and_mul: bool
) -> int:
    if per_token_activation and not is_act_and_mul:
        return 512
    return 256
```

## 性能声明

PR 正文 Test Result 给出在 `Nemotron-3-Super-120B-A12B-NVFP4`、2 张 GB200、TP2 上 static NVFP4 与 per-token NVFP4 的对比：GSM8K accuracy 93.56 ± 0.32 % → 93.93 ± 0.11 %；End-to-end output throughput 2254.7 → 2199.9 tokens/s；Evaluation wall time 60.20 s → 61.23 s；Mean TTFT 515.7 ms → 529.1 ms；Mean TPOT 37.76 → 38.35 ms/token。可见 per-token 变体精度略升、吞吐与延迟略降。测试条件为模型 Nemotron-3-Super-120B-A12B-NVFP4、硬件 2×GB200、TP2、GSM8K 评测；正文未注明除上述外的精度与 workload 细节。

## 验证与回归风险

正文 Test Plan 列出 kernel 级测试 `tests/quantization/test_modelopt.py`、`tests/kernels/moe/test_trtllm_nvfp4_moe.py` 与端到端 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4-per-token.yaml`。diff 改动集中在 NVFP4 MoE 的 ModelOpt/FlashInfer TRTLLM 量化路径与对应测试配置；static NVFP4 仍走 256 对齐，只有 per-token 非 gated 才改 512，hidden 对齐变化会影响权重 padding（hidden_dim_unpadded）。E2E 结果来自 GB200/Blackwell + Nemotron，正文未给出 ROCm 或其它硬件、其它量化与激活后端的回归评估，也未注明除上述对比外的影响范围。

## 关联

PR 正文明确提及所接入的 fused kernel 来自 FlashInfer 上游 flashinfer-ai/flashinfer#4618 与 #5031；vLLM 仓库内同分类前序/后续 PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
