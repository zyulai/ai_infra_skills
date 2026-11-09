# PR #54111: [Bugfix] Remove race in fused groupwise RMSNorm quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-08-28 |
| 分类 | Quantization |
| Label | bug, ready, ci-failure, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/54111 |

## 一句话结论

修掉 fused groupwise RMSNorm 量化归约的 shared-memory race,顺带 kernel 快 15–19%。

## 动机

H200 kernel CI 上出现三个 flaky 失败。其中两个 group-64 scale 失败是真实缺陷:`warpReduceMaxSpecialized` 的 shared-memory 归约中,每一级 reduction 读取上一级由其他 lane 写入的值却没有同步,CUDA racecheck 对失败特化报告数千 hazards。第三个 CUDA BF16 per-token FP8 case 的 scale 逐位一致,2,103,296 个输出中仅 1 个恰好差一个 E4M3 ULP——是合法的边界舍入,不是 race,需要测试侧区别对待。

## 优化点

把 shared-memory 归约替换为寄存器 warp shuffle:`warpReduceMax` 用 `VLLM_SHFL_DOWN_SYNC` 折半归约,只把最终 lane-0 结果写回 shared memory;同时覆盖 CUDA warp-32 与 ROCm wave-32/wave-64,并正确处理贡献者多于/少于 warp 宽度的组(组内跨 warp 步长读取后 shuffle 归约)。`compute_dynamic_per_token_scales` 的两处调用点(INT8/FP8 路径)同步更新。测试侧保持 scale 严格比较,但对 CUDA/BF16/per-token/FP8 这一特定配置允许有界数量的孤立 1-ULP 输出差(上限为每 100,000 元素 8 个)。这是 bugfix,归约树简化同时带来附带性能收益。

## 关键改动

```text
csrc/libtorch_stable/quantization/fused_kernels/layernorm_utils.cuh: warpReduceMaxSpecialized → 寄存器 shuffle 的 warpReduceMax(两处调用点)
tests/kernels/core/test_fused_quant_layernorm.py: CUDA BF16 per-token FP8 允许有界 1-ULP 输出差
```

```c
__device__ float warpReduceMax(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val = fmaxf(val, VLLM_SHFL_DOWN_SYNC(val, offset));
  return val;
}
```

## 性能声明

H200 直接测量完整 fused op(preallocated outputs);cold L2 显式 flush,hot 为每 CUDA graph replay 10 kernel,新旧交替构建控制漂移:2048x1024 group-64 INT8 cold 22.82→19.39 µs(**15.0% faster**)、hot 17.81→14.36 µs(**19.3% faster**);group-64 FP8 cold **14.4%**/hot **17.9%**;group-128 FP8 cold **11.0%**/hot 14.2%;小 batch(1x1024、1x5120)1.9–6.3%。注:CUPTI Python 本地不可用,FlashInfer benchmark helper 用了 CUDA-event/CUDA-graph fallback。

## 验证与回归风险

三个原始失败 case `3 passed`;附加 FP8 group-64 1x5120、INT8 group-128 1x5120 通过;100 次重复 2048x1024 group-64 scale 比较全部逐位确定;CUDA racecheck 0 hazards(修复前数千)。模型评测:seeded 两层 `Qwen2ForCausalLM` 将五个 RMSNorm 输出替换为 FP8 group quantize/dequantize(8x32 token batch):group 64/128 的 top-1 一致率 100%、max/mean abs logit diff 0/0、cosine 1.0、重复逐位确定。风险:改动为共享 header 的归约原语,影响所有 fused quant RMSNorm 配置(CUDA/ROCm、INT8/FP8、group-64/128、per-token);测试放宽仅限特定 CUDA BF16 per-token FP8 配置,其他配置仍严格。

## 关联

排查过相邻 open PR(均与本修不重叠):#45770(混合 input/weight dtype)、#49639(scalar RMSNorm 中间舍入边界)、#37196(旧静态量化 kernel 重构)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
