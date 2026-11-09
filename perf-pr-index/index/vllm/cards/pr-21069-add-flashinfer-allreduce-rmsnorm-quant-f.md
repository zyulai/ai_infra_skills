# PR #21069: Add FlashInfer allreduce RMSNorm Quant fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-07-31 |
| 分类 | Quantization \| tier: major |
| Label | performance, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 1 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/21069 |

## 一句话结论

集成 FlashInfer allreduce+RMSNorm+量化融合，B200 上 NVFP4 TPOT 较 custom ops 提升 10-15%。

## 动机

PR 正文 Purpose 说明新增 FlashInfer fused allreduce RMSNorm 与 Quantization 的集成，通过 `-O '{"pass_config": {"enable_fi_allreduce_fusion": true}, "level":3, "custom_ops":["+quant_fp8","+rms_norm"]}'` 启用。正文未说明改动前的具体瓶颈或问题。

## 优化点

在 `vllm/compilation/collective_fusion.py` 的图融合 pass 中新增 `STATIC_FP8_QUANT_OP` 与 `STATIC_FP4_QUANT_OP` 两个算子匹配项，使融合能识别量化节点。将 world_size=2 时 FlashInfer fused allreduce 的 `_FI_MAX_SIZES` 上限由 1MB 提升到 64MB。`call_trtllm_fused_allreduce_norm` 新增 `pattern_code`、`fuse_rms_quant`、`quant_out`、`scale_out`、`scale_factor` 参数，`pattern_code` 由固定的 `kARResidualRMSNorm` 改为按匹配结果传入，并把 FP4 量化 layout 固定为 `SWIZZLED`；非 FlashInfer 回退路径增加 fused rms norm static fp8 quant 融合分支。`vllm/config.py` 将 `fi_allreduce_fusion_max_token_num` 由 1024 改为 16384。

## 关键改动

```text
vllm/compilation/collective_fusion.py: 新增 FP8/FP4 量化算子匹配、放宽融合尺寸上限、按 pattern_code 传入量化输出与 SWIZZLED layout
vllm/config.py: fi_allreduce_fusion_max_token_num 默认 1024 → 16384
tests/compile/test_fusion_all_reduce.py: 引入 FixFunctionalizationPass、NoOpEliminationPass、QuantFP8 扩展融合测试
.buildkite/test-pipeline.yaml: 将 test_fusion_all_reduce.py 加入编译测试流水线
tests/utils.py: 测试配套调整
```

```text
STATIC_FP8_QUANT_OP = torch.ops._C.static_scaled_fp8_quant.default
STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.default
...
        2: 64 * MiB,  # 64MB
...
                pattern_code=pattern_code,
                quant_out=quant_out,
                scale_out=scale_out,
                layout_code=flashinfer_comm.FP4QuantizationSFLayout.SWIZZLED,
                scale_factor=scale_factor,
```

## 性能声明

Client `vllm bench serve --dataset-name sonnet`，Input len 550、Output len 150、B200 GPUs。FP8（RedHatAI/Meta-Llama-3-8B-Instruct-FP8 TP=2）：正文结论 "FP8 quantization PR improves TPOT compared to Custom ops by ~7-8%, no speedup compared torch.compiled (default) ops. Increases TTFT by up to 8%"。NVFP4（RedHatAI/Qwen3-32B-NVFP4 TP=2）：正文结论 "PR improves TPOT by up to 5% compared to torch.compiled ops, TTFT by up to 5%. Comparing to custom ops, PR improves TPOT by up to 10-15%, TTFT by up to 7%"。lm_eval gsm8k 精度：FP8 exact_match 0.7460/0.7483 → 0.7513/0.7528，FP4 0.6634/0.7468 → 0.6687/0.7559。

## 验证与回归风险

PR 正文给出 lm_eval gsm8k 校验命令与 FP8、NVFP4 两组精度结果；diff 新增 `tests/compile/test_fusion_all_reduce.py` 并写入 `.buildkite/test-pipeline.yaml`。风险方面正文列出 FP8 相对 torch.compiled 默认算子无加速且 TTFT 最多增加 8%，收益条件为启用该融合开关；正文未说明对其他模型、硬件或精度路径的影响。正文 Validation 还给出 lm_eval 命令（`--model local-completions ... --tasks gsm8k --num_fewshot 5 --batch_size 200`），并在 FP8、FP4 两组列出 gsm8k exact_match 的 baseline 与 PR 结果；Profiling 章节附有 NVFP4 模型（RedHatAI/Qwen3-32B-NVFP4 TP=2）的 baseline 与 PR 截图链接。正文 Benchmarks 还分别给出 FP8 与 NVFP4 在 QPS 1、5、10 三档下 baseline、fused custom ops、non-fused custom ops 与 PR 四类配置的 Mean/Median TTFT 与 Mean/Median TPOT 完整表格，用以对比融合收益与 TTFT 回退。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
