# PR #46070: [CI Bug] Revert #42379 to fix CI `Multi-Modal Models (Extended Generation 1)`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46070](https://github.com/vllm-project/vllm/pull/46070) |
| 作者 | yewentao256 |
| 合并日期 | 2026-06-18 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#42379 改动 RMSNorm 内核精度顺序致多模态音频生成 CI 失败;回退恢复测试通过。

## 问题现象
CI 作业 `Multi-Modal Models (Extended Generation 1)` 失败(正文链接 Buildkite build 72702)。本地复现命令 `pytest tests/models/multimodal/generation/test_audioflamingo3.py::test_single_and_batched_generation_match -xvs`:修复前 "Originally, error like CI";修复后 `1 passed, 17 warnings in 39.02s`。

## 根因
正文未披露机制性根因,只说回退 #42379 以修复该 CI。从 diff 看,#42379 曾改变 RMSNorm 及 RMSNorm+FP8 静态量化内核的舍入方式:由"先把归一化结果舍入到 scalar_t、再在权重原生 dtype 下相乘"改为"在 float 中乘完权重后一次性舍入";diff 注释自述这让融合路径严格更精确,从而在 FP8 E4M3 量化的 tie 边界处与未融合复合路径不一致(由 diff 推断)。

## 修复方案
回退 #42379 的内核改动:`csrc/libtorch_stable/layernorm_kernels.cu` 的 `rms_norm_kernel` 与 `fused_add_rms_norm_kernel`,以及 `csrc/libtorch_stable/layernorm_quant_kernels.cu` 的 `rms_norm_static_fp8_quant_kernel` 与 `fused_add_rms_norm_static_fp8_quant_kernel`,均恢复"先以 scalar_t 舍入归一化值、再乘权重"的旧精度路径。

```diff
-      scalar_t normalized = static_cast<scalar_t>(x * s_variance);
       if constexpr (HasWeight) {
-        dst.val[j] = normalized * src2.val[j];
+        float w = static_cast<float>(src2.val[j]);
+        dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
       } else {
-        dst.val[j] = normalized;
+        dst.val[j] = static_cast<scalar_t>(x * s_variance);
       }
```

(此摘录为回退后恢复的旧写法对应的 diff 行)

## 影响与回归风险
- 影响窗口:正文未写(自 #42379 合入起,本 PR 为其回退)。
- 回归面:libtorch_stable 下 RMSNorm 与 RMSNorm+FP8 静态量化内核的数值精度路径恢复为旧舍入顺序;融合内核与未融合复合路径在量化边界处重新保持一致;正文未提及性能影响。

## 关联
- #42379(被回退的 PR)
- CI 失败记录:https://buildkite.com/vllm/ci/builds/72702#019ed6b9-5522-4391-9f85-3b8699e878f9

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
