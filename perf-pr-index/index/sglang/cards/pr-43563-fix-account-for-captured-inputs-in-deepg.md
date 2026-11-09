# PR #43563: [Fix] Account for captured inputs in DeepGEMM masked workspace budget

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-10-13 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | quant, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/43563 |

## 一句话结论

修复 DeepGEMM masked 工作区预算估算漏算 CUDA graph capture 期间仍存活的输入与 scales,消除 prefill capture 阶段 OOM,比 forced-compact 规避方案快约 12.4%。

## 动机

DeepGEMM 的自动 standard-layout 策略可能在 prefill CUDA Graph capture 时选中一个会 OOM 的 masked workspace。原估算假设 runner 输入在 gate/up GEMM 后即被 `dispose_tensor` 释放,但 capture 期间 `dispose_tensor` 变成 no-op:输入及其 scales 在 activation 与 down GEMM 期间仍存活,实际峰值被低估。H200 默认内存占比/prefill chunk/graph 设置下可复现:Qwen3-30B-A3B-FP8 TP1 在 prefill shape 4608 时 masked 预算 3.15 GiB 下 `down_output` 需 2.38 GiB 而仅剩 1.20 GiB;Qwen3.5-122B-A10B-FP8 TP2 在 M=3072 时预算 5.37 GiB 需 4.88 GiB 仅剩 1.89 GiB。最小复现:`--model-path Qwen/Qwen3-30B-A3B-FP8 --tp-size 1 --moe-runner-backend deep_gemm`(30B 失败在热编译缓存下仍复现)。

## 优化点

修改 `_estimate_masked_standard_layout_peak_bytes`:不再取"三个阶段各自的峰值",而是把保留的输入行、输入 scales、down 输入行、down scales 全部计入,再叠加 `max(gateup_row_bytes, down_output_row_bytes)`,即按 capture 期间输入全程存活的最坏情形估峰值;warmup 与 capture 使用同一保守边界,保证两阶段 layout 选择一致。显式 layout 覆盖与预算占比不变。副作用是自动选择可能在其他模型上更早切到 compact:397B 形状 fixture 在 42.5 GiB 预算下 8K 即选 compact(4K 仍 masked),选择边界附近的性能可能变化。修复后 fixed-auto 保留 masked decode、大 prefill 用 compact,兼得两端。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py: masked standard-layout 峰值估算计入 capture 存活的输入与 scales
test/registered/kernels/ops/moe/test_minimax_quant_scatter.py: 策略测试参数化扩展两种失败几何与 packed/unpacked scales
```

```python
+    # Graph capture skips dispose_tensor, so the runner input remains live
+    # through activation and the down GEMM.
+    peak_row_bytes = (
+        input_row_bytes
+        + input_scale_row_bytes
+        + down_input_row_bytes
+        + down_scale_row_bytes
+        + max(gateup_row_bytes, down_output_row_bytes)
+    )
```

## 性能声明

Qwen3-30B-A3B-FP8,单 H200,320 请求、标称 8K 输入/1K 输出、并发 64,正反两序重复:

- fixed-auto:输出 2083.69/2083.45 tok/s,median TPOT 26.780/26.779 ms。
- forced compact(修复前的规避手段):1852.52/1853.72 tok/s,median TPOT 30.608/30.596 ms。
- fixed-auto 比 forced-compact 快约 12.4%;两种配置 median TTFT 均约 4.04 s。
- 默认 Triton runner 更快(2140.95 tok/s):本 PR 是修复 capture 可靠性,而非证明 DeepGEMM 更快;Qwen3.5-122B 用 DeepGEMM 仍慢于默认 Triton runner。

## 验证与回归风险

- 扩展 `test_minimax_quant_scatter.py`:两个失败几何(128/2048/768 @3.15 GiB 与 256/3072/512 @5.37 GiB)加 packed/unpacked scales 参数化,六个用例对原估算器会失败;GB300 上 53 passed。两模型在 H200 上完成启动、graph capture、serving 与 GSM8K。
- GSM8K(1314 题、greedy、2048 token 上限、同 case ID/prompt hash):Qwen3-30B fixed-auto 两轮 1215/1217,forced-compact 对照 1223/1221(反序对 lose 13/gain 9,不能证明数值等价);Qwen3.5-122B fixed-auto 1275 vs 默认 Triton 1277。原 DeepGEMM auto 因启动失败无法提供精度基线。
- 回归面:所有走 DeepGEMM auto layout 的模型,选择边界附近可能从 masked 切到 compact;CI 显示 Extra 与 AMD ROCm 10 失败,Base 通过。

## 关联

- 同区域:DeepGEMM MoE runner 与 CUDA graph capture 系列;同分类 CUDA Graph / Compiler 见 #40556。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
