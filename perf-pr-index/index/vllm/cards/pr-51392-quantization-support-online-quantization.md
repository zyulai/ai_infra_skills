# PR #51392: [Quantization] Support online quantization with partially pre-quantized checkpoints

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-09-04 |
| 分类 | Quantization \| tier: major |
| Label | documentation, speculative-decoding, ready, quantization, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 13 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/51392 |

## 一句话结论

让部分预量化 checkpoint 中未量化的 linear/moe 层也能叠加 online quantization，无需另发模型。

## 动机

PR 正文指出，此前无法对部分预量化 checkpoint 里保持未量化的 `linear`、`moe`（或细粒度 `targets`）层再做 online quantization，无论 checkpoint 用的是 Quark、ModelOpt、compressed-tensors 等任意 `quant_method`。原 `quant_method` 仍负责其已预量化的层，composed config 只在 checkpoint 返回未量化方法时应用请求的 online 方法。这放宽了为不同预设预量化层 / 不同 ignore list 分别发布多个模型的需求，正文举例 `nvidia/Qwen3.5-122B-A10B-NVFP4`（modelopt）与 `amd/Qwen3.5-397B-A17B-MXFP4`（quark）等仅量化非共享 MOE expert 的 checkpoint。

## 优化点

Online quantization 存为 `QuantizationConfig.online_quant_config`；`get_effective_quant_method` 解析全局 quant method，同时含 base checkpoint 方法与 online 方法。原 `QuantizationConfig` 子类实例（如 `Fp8Config`）被保留，模型开发者无需感知组合——正文说明这是为维持既有模型特化行为：Quark Qwen3Next 的 shared-expert fusion 读取 Quark 排除元数据、Kimi K2.5 量化多模态投影前检查 `CompressedTensorsConfig`、GLM4.1V 选 compressed-tensors 专用 QKV prefix、Qianfan OCR 应用 `Fp8Config` 的 visual-layer safety exclusions。被选中做 online 但已预量化的层会清晰报错而非静默替换；activation-only overrides 仍归原 checkpoint quantizer；`quantization_config.ignore` 只作用于 online。行为变化：`--quantization-config` 现可与 checkpoint 的 `quant_method` 一起用于 online weight composition。

## 关键改动

改动集中在量化配置解析与各层分发路径，并配套文档与测试。

```text
vllm/config/quantization.py: 引入 online_quant_config 与 get_effective_quant_method，组合解析 base + online 方法
vllm/model_executor/layers/linear.py: 线性层改用 effective quant method 分发
vllm/model_executor/layers/fused_moe/routed_experts.py: MoE 层接入组合后的量化方法
tests/quantization/test_online.py: 新增部分预量化 checkpoint 兼容性与选中层冲突测试
docs/features/quantization/online.md: 新增"在部分量化 checkpoint 的未量化层上做 online quantization"文档
```

新增文档片段（摘自 online.md 的 diff）：

```diff
+### Online quantization on unquantized layers from partially-quantized checkpoints
+The checkpoint `quant_method` remains responsible for its quantized layers,
+while the selected unquantized layers use the requested online method.
```

## 性能声明

PR 正文未给出定量性能数据，pack 的 "perf claims detected in body (regex, unverified)" 亦标注 "(none detected by regex)"；正文没有 benchmark 数字，也无模型 / 硬件 / 精度 / workload 等测试条件。

## 验证与回归风险

Test Plan 在 `tests/quantization/test_online.py` 新增 3 个 pytest：`test_online_prequantized_compatibility`、`test_online_prequantized_checkpoint_selected_layer_fails`、`tinysmokeqwen3moe`，覆盖 Quark / ModelOpt / compressed-tensors 分发、activation-only overrides、online ignore 保留 checkpoint quantization，以及真实 tiny compressed-tensors checkpoint 对选中 linear 与 MoE 层的冲突。Test Result 仅写 "Passes on MI355"。回归风险：正文 "Left to do" 明确列出 Real model testing 尚未完成，以及 `--quantization-config` 用于不同 `quant_method`、`QuantizationConfig.online_quant_config` 命名仍待确认；行为变化（选中已预量化层改为清晰失败）也会影响既有用法。

## 关联

正文明确线索：本 PR supersedes #49313（针对 amd/GLM-5.2-MXFP4 只量化 MOE 层）与 #48983（modelopt checkpoint）；合并后可配合 #51285；`shared_experts` online quantization + fusion 不在本 PR，另见 draft fxmarty-amd/vllm#6。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
