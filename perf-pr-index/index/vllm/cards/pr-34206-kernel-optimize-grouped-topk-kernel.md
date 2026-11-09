# PR #34206: [Kernel] Optimize grouped topk kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-02-20 |
| 分类 | MoE（tier: major） |
| Label | performance, ready, deepseek, model-bash |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 3 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/34206 |

## 一句话结论

新增面向小专家数的 grouped_topk kernel，H200 上 DeepSeek-R1 e2e output tok/s 由 669.51 升到 710.01（+5%）。

## 动机

正文 Purpose：本 PR 新增针对小专家数（单组 `num_experts<=512`，或多组 `num_experts<=256`）优化的 grouped topk kernel，覆盖 DeepSeek（num_experts=256, n_group=8）、Kimi-K2.5（384, n_group=1）、Nemotron（512, n_group=1）等；大专家数仍回落到既有 `grouped_topk_fused_kernel`。Profiling 发现原 `grouped_topk_fused_kernel` 在 H200 上约 7µs。

## 优化点

新增 `grouped_topk_fused_small_expert_count_kernel`：`csrc/moe/grouped_topk_kernels.cu` 引入 `moeTopKFuncs.cuh`，并增加面向各模型专家数的编译期常量（`NumNemotronExperts=512`、`NumKimiK2Experts=384`、`NumDeepseekExperts=256`、`MaxSupportedExpertCount`、`MaxNumExpertsUnit=128`、`NumTopGroupScores=2`、`DefaultMaxNumTopExperts=8`、`MaxSupportedTopExperts=22`、`MaxNumTopGroups=4`），在小专家数场景用更专用的实现替代通用 kernel；`csrc/moe/moeTopKFuncs.cuh` 为共用函数头，`tests/kernels/moe/test_grouped_topk.py` 覆盖新 kernel。正文未展开说明具体算法差异；Profiling 显示新 kernel 约 4µs。

diff 中 `csrc/moe/grouped_topk_kernels.cu` 还把文件头标注的 TensorRT-LLM 参考版本由 `v0.21.0` 更新为 `v1.3.0rc2`，新增 `#include "moeTopKFuncs.cuh"` 与 `#include <cmath>`，并把 `WARP_SIZE` 由 `constexpr int32_t WARP_SIZE = 32` 改为 `static constexpr int`；新 kernel 的实现位于 `@@ -657,76 +669,335 @@` 对原 `grouped_topk_fused_kernel` 区域的扩展处。正文未说明 `moeTopKFuncs.cuh` 内的具体算法，也未给出新旧 kernel 的分块参数对比。Accuracy Testing 的 `python3 tests/evals/gsm8k/gsm8k_eval.py` 还给出 Output tokens per second：DeepSeek-R1 384.980→429.395、Kimi-K2.5 483.480→494.743、Nemotron 900.533→1014.436，Questions per second 分别为 3.930→4.360、4.875→4.989、4.098→4.627。Profiling 段明确 "Profiled DeepSeek-R1 model on H200" 并附 Main/PR 两张截图；DeepSeek-R1 的 serve 命令带 `--tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 16 --trust-remote-code --no-enable-prefix-caching`，bench 命令带 `--sharegpt-output-len 300 --max-concurrency 16 --num-prompts 1000 --num-warmups 50 --ignore-eos`；Kimi-K2.5 复用同一 bench 配置并加 `--mm-encoder-tp-mode data`，Nemotron 的 serve 改为 `--tensor-parallel-size 1`。

## 关键改动

```text
csrc/moe/grouped_topk_kernels.cu: 新增 grouped_topk_fused_small_expert_count_kernel 及编译期常量
csrc/moe/moeTopKFuncs.cuh: topk 公共函数头
tests/kernels/moe/test_grouped_topk.py: grouped topk 单元测试
```

```diff
+static constexpr int NumNemotronExperts = 512;
+static constexpr int NumKimiK2Experts = 384;
+static constexpr int NumDeepseekExperts = 256;
+static constexpr int MaxSupportedExpertCount =
+    std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
+static constexpr int MaxNumExpertsUnit = 128;
+static constexpr int NumTopGroupScores = 2;
+static constexpr int DefaultMaxNumTopExperts = 8;
+static constexpr int MaxSupportedTopExperts = 22;
+static constexpr int MaxNumTopGroups = 4;
```

## 性能声明

测试条件：H200（Profiling "Profiled DeepSeek-R1 model on H200"），Main `grouped_topk_fused_kernel` 约 7µs、PR `grouped_topk_fused_small_expert_count_kernel` 约 4µs。Benchmarking 三组（sharegpt、sharegpt-output-len 300、max-concurrency 16、num-prompts 1000、num-warmups 50、--ignore-eos）：

| Model | 配置 | Output tok/s Main → PR | Total tok/s Main → PR | Mean TPOT (ms) Main → PR | 正文结论 |
|---|---|---|---|---|---|
| DeepSeek-R1-0528 | tp8, EP, max-num-seqs 16 | 669.51 → 710.01 | 1181.50 → 1252.97 | 22.93 → 21.61 | e2e output token throughput improves 5% |
| moonshotai/Kimi-K2.5 | tp8, EP, mm-encoder-tp-mode data | 794.08 → 806.44 | 1394.03 → 1415.73 | 19.00 → 18.73 | e2e output token throughput improves 1% |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | tp1 | 1328.81 → 1344.11 | 2374.30 → 2401.63 | 11.47 → 11.36 | e2e output token throughput improves 1% |

（Mean TTFT：DeepSeek-R1 265.93→253.94 ms、Kimi 324.19→314.71 ms、Nemotron 158.18→151.62 ms；perf claims 段标注 5%、1%。）

## 验证与回归风险

验证：`pytest -s -v tests/kernels/moe/test_grouped_topk.py`（正文称 Unit tests passed）；Profiling 截图对比；上述三组 serving benchmark；Accuracy Testing `python3 tests/evals/gsm8k/gsm8k_eval.py`，DeepSeek-R1 Accuracy 0.955→0.955、Kimi-K2.5 0.939→0.939、Nemotron 0.265→0.279（Invalid responses 0.001→0.000）。可能影响路径：小专家数场景改用新 kernel，大专家数回落既有 `grouped_topk_fused_kernel`；PR 未注明其他回归风险。Profiling 段还说明 Main 的 `grouped_topk_fused_kernel` 与 PR 的 `grouped_topk_fused_small_expert_count_kernel` 分别对应约 7µs 与约 4µs 的两张截图对比；正文 Test Result 段仅写 "Unit tests passed"，未列出更细的逐项数值，也未说明新 kernel 在 `num_experts` 超过 512 时的具体回退判定位置。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
