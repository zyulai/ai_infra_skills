# PR #23274: [Kernel] Add fused grouped_topk kernel for MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2025-08-25 |
| 分类 | MoE (tier: major) |
| Label | rocm, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 8 个文件(CUDA kernel、CMake、ops 绑定、fused_moe 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/23274 |

## 一句话结论

为 MoE 新增融合 grouped_topk kernel,DeepSeek-R1 输出 token 吞吐约 +9.7%。

## 动机

PR 正文 Purpose:本 PR 为 MoE 增加 fused grouped_topk kernel。在 `grouped_topk_kernels.cu` 中新增 `renormalize` 参数(不再总是 renormalize),并给 `grouped_topk` 与 `fused_grouped_topk` 都引入 `routed_scaling_factor` 参数(grouped topk 计算所需)。正文未展开更细的瓶颈描述。

## 优化点

新增 CUDA kernel `grouped_topk`(csrc/moe/grouped_topk_kernels.cu),把 group 打分、组内 topk、renormalize 与 routed scaling 融为一次 `ops.grouped_topk` 调用;`vllm/model_executor/layers/fused_moe/fused_moe.py` 的 `grouped_topk()` 在 `VLLM_USE_FUSED_MOE_GROUPED_TOPK` 开启、平台为 CUDA、num_expert_group<=32、topk<=32 且 e_score_correction_bias 非空时改走新的 `fused_grouped_topk`,否则保留原 PyTorch 分解路径。新路径把 `routed_scaling_factor` 直接乘进 topk_weights,CUDA 侧对应加 `renormalize` 参数。配套改动 CMakeLists.txt、moe_ops.h、torch_bindings.cpp、vllm/_custom_ops.py、vllm/envs.py 接通算子与开关,并新增 tests/kernels/moe/test_grouped_topk.py。

## 关键改动

```text
csrc/moe/grouped_topk_kernels.cu: 新增融合 grouped_topk CUDA kernel,带 renormalize 参数
vllm/model_executor/layers/fused_moe/fused_moe.py: grouped_topk() 条件分派到新 fused_grouped_topk,引入 routed_scaling_factor
CMakeLists.txt / csrc/moe/moe_ops.h / csrc/moe/torch_bindings.cpp / vllm/_custom_ops.py / vllm/envs.py: 编译、算子声明、绑定与开关
tests/kernels/moe/test_grouped_topk.py: 新增单测
```

关键机制代码(条件分派与新入口):

```python
    if envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and \
            current_platform.is_cuda() and \
            num_expert_group <= 32 and topk <= 32 and \
            e_score_correction_bias is not None:
        return fused_grouped_topk(...)
...
def fused_grouped_topk(...):
    scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
    topk_values, topk_indices = ops.grouped_topk(
        scores, scores_with_bias.to(scores.dtype), num_expert_group,
        topk_group, topk, renormalize, routed_scaling_factor)
    return topk_values.to(torch.float32), topk_indices.to(torch.int32)
```

## 性能声明

PR 正文 Benchmarking:在 p5e.48xlarge(8× H200)上 `vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 8 ... --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'`,压测 `python3 benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-R1 --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --max-concurrency 8`。结果 output token throughput 285.02 → 312.81 tok/s、request throughput 1.64 → 1.85 req/s、total token throughput 644.27 → 718.26 tok/s、mean TPOT 26.93 → 24.52 ms,正文称 around ~9.7% improvement on output token throughput。精度(lm_eval gsm8k 5-shot,DeepSeek-R1):baseline 0.9538/0.9538 → 本 PR 0.956/0.956(strict/flexible)。正文 This PR 的 Mean TTFT 记为 281563.90 ms(baseline 147.14 ms),量级差异极大,素材如此照抄、不作推断。

## 验证与回归风险

Test Plan:`pytest -s -v tests/kernels/moe/test_grouped_topk.py`(正文称 unit tests passed),并有 lm_eval gsm8k 精度测试与上述 serving benchmark。影响路径:开启 `VLLM_USE_FUSED_MOE_GROUPED_TOPK` 且满足 CUDA / group<=32 / topk<=32 / 有 bias 条件时走融合 kernel 的 MoE routing(DeepSeek-R1 等);不满足条件与 CPU/ROCm 等平台仍走原分解路径,gsm8k 精度基本持平(0.9538 → 0.956)。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
