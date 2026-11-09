# PR #32616: [JIT] Restore the previous division behavior in per-token group quantization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1e4ves |
| 合并 | 2026-07-28 |
| 分类 | Speculative Decoding | tier: major |
| Label | bug, performance, quant, ready-for-review, run-ci, nvidia, hopper, jit-kernel, run-ci-extra, kernel |
| Diff | diffstat 缺失(素材未提供统计数字);diff 仅 1 个文件、1 行实质改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/32616 |

## 一句话结论

恢复 per-token-group 量化中 `kMaxValue / amax` 直接除法,修复 Hopper 上 FP8 舍入差异导致的 MTP 接受长度回退。

## 动机

#30924 把 FP32-scale 量化乘子从直接除法改成了显式倒数乘法(`kMaxValue * __frcp_rn(amax)`),这在 Hopper 上引入了微小的 FP8 舍入差异,与此前 v2 kernel 的输出不再逐位一致。该差异被关联到 #32582 报告的 GLM-5.2 decode 回归。PR 作者明确说明确切机制尚不清楚,且此改动不一定能完全解决该回归,但先把共享路径恢复为与 v2 kernel 一致的行为。

## 优化点

这是一个数值一致性修复而非算法改动:JIT per-token-group 量化 kernel 的共享路径里,量化 scale 的计算由"先用 `__frcp_rn` 求 amax 的近似倒数、再与 kMaxValue 相乘"改回"直接计算 `kMaxValue / amax`"。两者在 IEEE 浮点语义下舍入路径不同(`__frcp_rn` 是舍入到最近偶数的倒数近似,与除法指令结果可差 1 ulp),放大到 FP8 e4m3 编码后会出现个别量化码与 scale 的偏差。恢复除法后,共享路径与旧 v2 kernel 逐位一致,消除该路径上由 #30924 引入的 FP8 输出漂移。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh: FP32-scale 路径量化 scale 计算恢复为直接除法
```

关键片段(改动本身):

```cuda
// fp32 scale: multiply in fp32 (hmul2 brings too much precision loss)
scale_inv = raw_scale;
- const float quant_scale = kMaxValue * __frcp_rn(amax);
+ const float quant_scale = kMaxValue / amax;
```

## 性能声明

PR 正文无吞吐/时延 benchmark,提供的是量化一致性与接受长度证据:

- FP32-scale CUDA 测试(H20):8 项全部通过。
- 与旧 v2 kernel 逐位对比:FP8 量化码与 scale 零 mismatch。
- 配合 #30393,DeepSeek-V3.2 MTP L3 replay accept length 从 2.462 恢复到 3.368(冷启动运行接受长度为 3.459,尚未完全回到该水平)。硬件为 Hopper 系(H20 用于 CUDA 测试),PR 未注明完整模型/精度/workload 条件。

## 验证与回归风险

验证依赖现有 FP32-scale CUDA 单测(H20 上 8 项通过)及与 v2 kernel 的 bitwise 对比。风险面:该 .cuh 是 JIT 量化 kernel 的共享头文件,改动影响所有走 per-token-group FP32-scale 量化路径的模型与 GEMM;对依赖 #30924 行为的场景属于有意回退,数值结果会与 main 上近期版本有极小差异(与更早版本一致)。PR 作者自述不保证完全修复 #32582 的 GLM-5.2 decode 回归,合并时 PR Test (Base) CI 显示为失败状态(Extra 为等待中),需关注后续验证。

## 关联

前序:#30924(引入 `__frcp_rn` 倒数乘法的改动,本 PR 部分回退它)、#30393(配合使用恢复 accept length)。问题来源:#32582(GLM-5.2 decode 回归)。同分类区域:MTP/投机解码精度相关改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
