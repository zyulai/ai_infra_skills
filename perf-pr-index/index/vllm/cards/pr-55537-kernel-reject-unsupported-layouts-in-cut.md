# PR #55537: [Kernel] Reject unsupported layouts in CUTLASS 3.x scaled_mm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jackLei0901 |
| 合并 | 2026-10-19 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, ci/build, nvidia |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/55537 |

## 一句话结论

CUTLASS 3.x scaled_mm 派发前拒绝非 packed/未对齐操作数，test_cutlass_subset 由 1 failed 转 1 passed。

## 动机

正文 Overview：在 CUTLASS 3.x scaled-mm 派发前拒绝不支持的 padded 操作数与 misaligned 指针（#55534），packed 输入与 CUTLASS 2.x 行为保持不变。Details 说明根因：C3x callers 从 shapes 重建 packed strides，不兼容的操作数会「silently address the wrong elements」，因此本改动选择在 dispatch 处拒绝而非新增 stride 支持；并强调「This PR passes the repro by rejecting the padded input, not by supporting it」，反向还原改动可在 main 复现失败。正文还称未识别出需要 padded 操作数的 in-tree serving caller。这是正确性修复，非提速。

## 优化点

`csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu` 新增 `check_c3x_scaled_mm_operands(c, a, b)`：对 A、B、output 逐一检查 leading stride（`size(dim) <= 1 || stride(dim) == size(1 - dim)`，保留 size-one leading-dimension 例外）与 16-byte 指针对齐（`reinterpret_cast<std::uintptr_t>(data_ptr()) % 16 == 0`），不满足则用 `STD_TORCH_CHECK` 报错。该检查被插入到 SM120、SM100、SM90 三个 `cutlass_scaled_mm` 分支以及 `cutlass_scaled_mm_azp` 的 SM90 分支派发之前；CUTLASS 2.x 路径与 packed 输入不变。`.buildkite/test_areas/kernels.yaml` 把 B200 selector 调整为 `-k 'fp8 or cutlass_subset'`（保留原 FP8 测试并加入已有 subset 回归），`tests/kernels/quantization/test_cutlass_scaled_mm.py` 更新该 subset 回归。正文 Claims 自述修订后 diff 为 +34/-1 across three files（含测试与 CI），但 pack 元信息未提供 diffstat，故表格按 pack 标注如实填写。

## 关键改动

```text
csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu: 新增 check_c3x_scaled_mm_operands 并在 SM120/SM100/SM90 与 azp 分支派发前调用
.buildkite/test_areas/kernels.yaml: B200 selector 改为 -k 'fp8 or cutlass_subset'
tests/kernels/quantization/test_cutlass_scaled_mm.py: 更新 test_cutlass_subset 回归

# 机制摘录 (scaled_mm_entry.cu)
void check_c3x_scaled_mm_operands(const torch::stable::Tensor& c,
                                  const torch::stable::Tensor& a,
                                  const torch::stable::Tensor& b) {
  auto check = [](const torch::stable::Tensor& t, const char* name, int dim) {
    STD_TORCH_CHECK(t.size(dim) <= 1 || t.stride(dim) == t.size(1 - dim),
                    "CUTLASS 3.x scaled_mm requires packed operands; ", name,
                    " is not packed");
    STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0,
                    "CUTLASS 3.x scaled_mm requires 16-byte-aligned pointers; ",
                    name, " is misaligned");
  };
  check(a, "A", 0);
  check(b, "B", 1);
  check(c, "output", 0);
}
```

## 性能声明

PR 正文未给出加速类定量性能数据，且明确标注 serving benchmark「these are not a speedup claim」。条件：H800 PCIe（SM90）、torch 2.13.0+cu130、main `3e8e1364ff` 源码构建、验证 `5b18e7656c` on base `3e8e1364ff`（在 #60721 源码布局重构之前）；模型 Meta-Llama-3-8B-Instruct、fp8、bf16、TP1、max-model-len 8192、seed 42。Accuracy（lm_eval GSM8K 5-shot，1319 例全完成）：main 与 Now 均为 flexible-extract `0.7475 ± 0.012`、strict-match `0.7460 ± 0.012`，sample JSONL byte-identical。Serving benchmark（`vllm bench serve`，random，input 1024/output 128，128 prompts，16 warmups，seed 42）rep2：main output tok/s `6616.12`、mean TTFT `264.56 ms`、P99 TTFT `472.46 ms`；Now `6556.76`、`281.78 ms`、`492.78 ms`。三次重复共九次日志，首轮约 3.0k tok/s、后续 6.47-6.66k tok/s，warm-state 差异原因未隔离。

## 验证与回归风险

Validation 命令 `.venv/bin/python -m pytest tests/kernels/quantization/test_cutlass_scaled_mm.py::test_cutlass_subset -v -s`：main `1 failed`（159110/262144 = 60.7% 元素不匹配，最大绝对差 9.75、允许 1.0），Now `1 passed`；反向还原可复现 main 失败。Historical H800 验证通过 7 个 boundary cases（含 6 个后按 yewentao256 要求从 committed tests 移除的 dedicated cases，日志留在 gist）、36 个 packed/output-dtype/CUDA-graph cases、11 个 offline INT8/AZP/size-one/pointer cases；每次 serving 记录 `Selected CutlassFP8ScaledMMLinearKernel`；仅 SM90 本地测试。风险：CUTLASS 3.x 现在会对 padded/misaligned 操作数报错（此前为静默错误），packed 输入与 CUTLASS 2.x 不变；新源码布局未在本地构建/GPU 测试，CUDA CI 将是首次构建验证；B200 CI 结果 pending；架构级 contract 仍需达成一致。

## 关联

PR 正文明确提及修复目标 #55534、源码布局重构 #60721、开放的 stride-support 替代方案 #56248 与 #56659、以及作者于 10 月 8 日关闭的 #56480（其 SM100 stride support 与 reject-only contract 冲突，正文强调这不代表 maintainer 批准）；并声明本 PR 是对既有 #55537 的修订而非并行修复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
