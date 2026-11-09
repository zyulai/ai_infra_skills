# PR #56214: [Model] Support DeepSeek-V4.1-Flash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-11 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | new-model, rocm, speculative-decoding, torch.compile, ci/build, multi-modality, tool-calling, deepseek, kv-connector, nvidia, quantization, mrv2, dflash, rust, kv-cache-manager, DSv4.1 |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾注明 "43 file(s) dropped from diff for size (see PR for full diff)" |
| 链接 | https://github.com/vllm-project/vllm/pull/56214 |

## 一句话结论

新增 DeepSeek-V4.1-Flash 模型支持（含 MXFP8/ROCm/MoE 测试与 CI 配置）；PR 正文为空，未给出定量性能数据。

## 动机

PR 正文为空（pack 中 PR body 段仅为一对空代码围栏），未提供改动目的、瓶颈或背景说明，按规则写「PR 未注明」。从 title 与 Label 仅可知这是一次新模型（DeepSeek-V4.1-Flash）支持改动。

## 优化点

PR 正文为空，未说明算法/内核/调度层面的优化机制，按规则写「PR 未注明」。从可见 diff 只能看到测试与 CI 层面的配套改动：`.buildkite/test_areas/kernels.yaml` 把 `kernels/test_mhc_kernels.py` 单独隔离运行（注释说明为避开 FlashInfer 的 import-time ptxas override）并在批量 kernels 命令中 `--ignore`；`tests/fusion/test_quant_activation_contract.py` 把 `_POSSIBLE_MXFP8_KERNELS` 与 `FlashInferCutedslMxfp8LinearKernel`、`FlashInferCutlassMxfp8LinearKernel` 纳入 kernel 类枚举；另新增/扩展 ROCm attention、`fused_q_kv_rmsnorm`（MXFP8）、MXFP8 linear、compressor KV cache、engram 等测试。pack 因体积丢弃 43 个文件，模型实现主体未在 diff 中展示。

## 关键改动

```text
.buildkite/test_areas/kernels.yaml: 隔离 test_mhc_kernels.py 并从批量 kernels 测试中排除
tests/fusion/test_quant_activation_contract.py: 把 MXFP8 kernel 类加入量化激活契约测试的枚举
tests/kernels/core/test_fused_q_kv_rmsnorm.py: 新增 Blackwell/SM100 下 MXFP8 fused_q_kv_rmsnorm_quant 与分别量化的一致性测试（含 CUDA graph replay）
tests/kernels/attention/test_rocm_triton_attn_dsv4.py / tests/kernels/quantization/test_rocm_mxfp8_linear.py: ROCm DSv4 attention 与 MXFP8 linear 相关测试
```

```diff
+    # Isolate mHC from FlashInfer's import-time ptxas override.
+    - pytest -v -s kernels/test_mhc_kernels.py
```

```diff
+@pytest.mark.skipif(
+    not current_platform.has_device_capability(100), reason="MXFP8 needs Blackwell"
+)
```

## 性能声明

PR 正文未给出定量性能数据（PR body 为空，perf claims 段标注 "(none detected by regex)"）；PR 未注明测试条件。

## 验证与回归风险

验证：可见 diff 新增/扩展多处测试——`tests/fusion/test_quant_activation_contract.py`（纳入 MXFP8 kernel 类）、`tests/kernels/core/test_fused_q_kv_rmsnorm.py`（SM100 上 MXFP8 量化一致性，含 CUDA graph replay）、`tests/kernels/attention/test_rocm_triton_attn_dsv4.py`（新增 `_get_cached_wo_a_bf16` 忽略 retained scale 的用例）、`tests/kernels/quantization/test_rocm_mxfp8_linear.py`、`tests/kernels/test_compressor_kv_cache.py`、`tests/kernels/test_engram.py`，以及 `.buildkite/test_areas/kernels.yaml` 的 CI 命令调整。PR 正文未注明测试结果与回归风险；pack 丢弃 43 个文件，无法确认其余改动内容。

PR 正文为空，pack 的 perf claims 段标注 "(none detected by regex)"；Label 中的 DSv4.1、mrv2、dflash、kv-cache-manager、rust、torch.compile 等仅为分类标签，未提供可引用的改动说明或性能目标。可见 diff 全部为测试与 CI 配置文件（`.buildkite/test_areas/kernels.yaml` 与 `tests/` 下的多个文件），未见模型实现源码；其中 `tests/kernels/core/test_fused_q_kv_rmsnorm.py` 的新用例对 `num_tokens` 取 0、1、17、128、129、1024，`q_size` 取 384、1280、1344，`dtype` 取 bfloat16/float16，并在 CUDA graph replay 下校验结果。

## 关联

无（PR 正文为空，diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
