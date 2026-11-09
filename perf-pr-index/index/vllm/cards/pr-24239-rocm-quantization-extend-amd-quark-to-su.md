# PR #24239: [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xuebwang |
| 合并 | 2025-11-11 |
| 分类 | Quantization（tier: major） |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, gpt-oss, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/24239 |

## 一句话结论

让 AMD Quark 的 layerwise {MXFP4, FP8} 混合精度量化模型可在 vLLM 加载推理。

## 动机

此前 vLLM 的量化模型多为单一 scheme（如 MXFP4、FP8 / PTQ 模型）。本 PR 扩展到 layerwise 混合精度量化模型推理：由 AMD Quark 为给定模型搜索每层配置再量化，本 PR 聚焦 {MXFP4, FP8} 混合（FP8 为 per-tensor scheme）。文档说明：单用设备支持的最低精度（如 AMD Instinct MI355 的 MXFP4、MI300 的 FP8）可最大化吞吐，但这类激进方案可能损害目标任务精度，混合精度可在 accuracy 与 throughput 间取平衡。

## 优化点

在 `quark.py` 的 `QuarkConfig.from_config` 放宽 kv_cache 配置校验：除 `kv_cache_set.issubset(layer_quant_set)` 外允许按 `fnmatch.fnmatchcase` 的模式匹配，并把 q_configs 由按 `kv_cache_group` 名字 `.get(name)` 改为遍历 `layer_quant_config.items()` 用 `fnmatch` 匹配收集；一致性比较由 `deep_compare(q_config, q_configs[0])` 改为只比较 `["output_tensors"]`。`_find_matched_config` 新增 `_matches_pattern`：模式不含 `*` 时用 `layer_name in pattern`，否则 `fnmatch.fnmatch`，并返回匹配到的 config。由此支持每层不同量化 scheme 的配置。文档 `docs/features/quantization/quark.md` 新增 AMP 章节、示例模型清单与 lm_eval 用法。

## 关键改动

```text
vllm/model_executor/layers/quantization/quark/quark.py: from_config 支持 layer 名 fnmatch 模式匹配、按 output_tensors 比较 kv_cache 量化一致性；_find_matched_config 新增 _matches_pattern 并返回匹配 config。
docs/features/quantization/quark.md: 新增 "Using Quark Quantized layerwise Auto Mixed Precision (AMP) Models" 章节、三个示例 AMP 模型与 lm_eval 命令。
tests/quantization/test_mixed_precision.py（新增 69 行）: lm_eval 精度测试，QUARK_MXFP4_AVAILABLE 要求 amd-quark>=0.8.99。
```

```text
            def _matches_pattern(layer_name, pattern):
                if "*" not in pattern:
                    return layer_name in pattern
                return fnmatch.fnmatch(layer_name, pattern)

            for name_pattern, config in layer_quant_config.items():
                if _matches_pattern(layer_name, name_pattern):
                    return config
```

## 性能声明

PR 正文未给出定量性能数据。Test Result 仅为一幅图片链接（`https://github.com/user-attachments/assets/acbce0ae-b8dc-4952-9b6f-7bb19fe09f90`），正文无可读 benchmark 数字，PR 未注明测试条件（模型 / 硬件 / 精度 / workload）。正文称"show the model accuracies on several commonly used tasks only using Quark emulation kernel for MXFP4 and triton kernel for FP8"；TODO 中 "benchmark hardware metrics" 仍为未勾选项。diff 的 `tests/quantization/test_mixed_precision.py` 给出的是精度期望值（非 PR 正文 benchmark）：`amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8` arc_challenge 0.52 / mmlu 0.72，`amd/Llama-2-70b-chat-hf-FP8-MLPerf-fp8_attn_quark_format` arc_challenge 0.53 / mmlu 0.61，`rtol=0.05`。

## 验证与回归风险

验证：新增 `tests/quantization/test_mixed_precision.py`，用 `lm_eval.simple_evaluate`（`model="vllm"`，`batch_size=8`，`tensor_parallel_size=4`、`dtype=auto`、`gpu_memory_utilization=0.8`）校验 arc_challenge / mmlu，skip 条件为 amd-quark 不可用；Test Plan 列出 3 个 AMP 模型（Llama-2-70b、Mixtral-8x7B、Qwen3-8B 的 `WMXFP4FP8-AMXFP4FP8-AMP-KVFP8`）。风险：TODO 明确 "extend model coverages"、"benchmark hardware metrics" 未完成；当前仅支持 {MXFP4, FP8} 混合，尚未支持 Unquantized Linear / MoE 层（{MXFP4, FP8, BF16/FP16}）与 MXFP6 方案；MXFP4 用 Quark emulation kernel、FP8 用 triton kernel。

## 关联

无（正文 TODO 提及后续支持 {MXFP4, FP8, BF16/FP16}、MXFP6 等混合 scheme，未给 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
