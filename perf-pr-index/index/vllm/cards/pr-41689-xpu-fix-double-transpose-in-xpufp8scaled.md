# PR #41689: [XPU] Fix double-transpose in XPUFP8ScaledMMLinearKernel for W8A8 quant method

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @libinta |
| 合并 | 2026-05-14 |
| 分类 | Quantization (tier: major) |
| Label | documentation, performance, new-model, rocm, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, deepseek, cpu, gpt-oss, kv-connector, nvidia, mistral |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/41689 |

## 一句话结论

修复 XPU FP8 线性核对 W8A8 量化方法权重造成的双重转置:ModelOpt/fp8 等 W8A8 在 Intel GPU 上从报错变为可用。

## 动机

XPU FP8 kernel 的 `process_weights_after_loading` 此前无条件对权重做 `.t()` 转置,但 W8A8 量化方法(modelopt、fp8)在调用该 kernel 之前已经用 `.t()` 把权重转置好,于是发生双重转置:输出维度错误,并触发 `split_with_sizes` 报错(例如 QKV split 期望 6144 却拿到 4096),W8A8 量化模型在 XPU 上无法运行。

## 优化点

修复以权重连续性为判据:checkpoint 权重永远是连续的 [out, in] 布局、需要转置;W8A8 方法预转置过的权重是非连续的 `.t()` 视图、应保持原样。具体实现比较 `weight.shape == (out_features, in_features)` 且(非方阵,或方阵时以 `is_contiguous()` 作 tie-breaker)才执行转置,否则视为已在 [in, out] 布局、no-op。量化算术本身不变。

## 关键改动

```text
vllm/model_executor/kernels/linear/scaled_mm/xpu.py: process_weights_after_loading 按形状+连续性条件转置,避免双重转置
.buildkite/intel_jobs/test-intel.yaml: Intel CI 增加 nvidia/Llama-3.1-8B-Instruct-FP8 ModelOpt W8A8 冒烟用例
```

核心修复:

```python
if weight.shape == (out_features, in_features) and (
    in_features != out_features or weight.is_contiguous()
):
    replace_parameter(layer, "weight", weight.data.t())
# else: already in [in, out] layout — no-op
```

## 性能声明

PR 正文无性能数字,属正确性修复(让 W8A8 在 XPU 上从崩溃变为可运行)。给出的验证均为功能通过:CT W8A16(`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`)、ModelOpt W8A8(`nvidia/Llama-3.1-8B-Instruct-FP8`)、ModelOpt W8A8 + FP8 KV cache + Triton attention 均通过。硬件等条件 PR 未注明(XPU/Intel GPU 为隐含目标平台)。

## 验证与回归风险

Test Plan/Result 章节为空,验证即上述三个手工用例;CI 侧在 Intel Buildkite 流水线中新增 `--quantization modelopt --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 4096` 的 Llama-3.1-8B-FP8 生成冒烟,防回归。风险面:改动只影响 `XPUFP8ScaledMMLinearKernel` 的权重后处理;方阵权重的 tie-breaker 依赖"checkpoint 权重连续、转置视图非连续"的约定,若上游传入其他形式的非连续 [out, in] 权重会被误判为已转置;CT W8A16(checkpoint 路径)行为不变。

## 关联

同分类相关:XPU 量化路径无其他卡片;可参考索引中 Quantization 分类下 W8A8 相关卡(如 pr-18778、pr-20071)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
