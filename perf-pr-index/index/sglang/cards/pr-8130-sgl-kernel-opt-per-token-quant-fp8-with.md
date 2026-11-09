# PR #8130: [sgl-kernel] Opt per_token_quant_fp8 with warp reduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-07-23 |
| 分类 | Quantization (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,改动集中在 1 个 CUDA 源文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/8130 |

## 一句话结论

重写 sgl-kernel 的 per_token_quant_fp8 内核:大 batch 下用 warp reduce 替代 CUB block reduce,拿到 5-7% 提速。

## 动机

per-token dynamic FP8 量化是 FP8 推理路径上的高频算子。原实现每个 CTA 处理一个 token、依赖 CUB 的 block reduce 求 absmax,共享内存归约开销在 batch 变大后成为瓶颈。PR 目标是在大 batch 场景下降低该内核耗时。

## 优化点

新内核改为 warp-local 组织:一个 warp(32 线程)负责一个 token,一个 256 线程 CTA 同时处理 8 个 token。求 absmax 用 `warpReduceMax` 的 shuffle 归约完成,不再走 CUB block reduce 的共享内存多轮归约;scale 直接写回、用 `1/scale` 做第二遍反量化写入,数据访问用 flashinfer 的 `vec_t<T,16>` 向量化 load/store。调度侧按 `num_tokens >= sm_count * 2 * TOKENS_PER_CTA`(sm_count 硬编码为 132)选择走新的 warp 路径,小 batch 仍回落到原 baseline 内核,避免小规模下退化。

## 关键改动

```text
sgl-kernel/csrc/gemm/per_token_quant_fp8.cu: 唯一改动文件。新增 per_token_quant_fp8_kernel(T, kTokensPerCTA=8, kVecSize=16) warp-local 版本,原实现保留为 per_token_quant_fp8_small_batch_kernel,入口按 token 数分流。
```

关键机制摘录:

```cuda
const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
...
for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
  ...
  max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
}
float warp_max = warpReduceMax(max_value);
```

## 性能声明

PR 正文给出 `bench_per_token_quant_fp8.py` 前后对比(SGL Kernel 列,单位未在 PR 中标注,数值应为耗时类指标,越小越快):修改前 batch 128 / seq 4096 为 7765.30,修改后同配置为 7294.50;batch 64 / seq 4096 从 3888.48 降到 3668.46;batch 32 / seq 4096 从 1953.95 降到 1857.89。作者总结为 5-7% speedup(大 batch)。对照组为 VLLM 实现,数值不变(如 batch 128/4096 为 9216.30 / 9214.90)。测试硬件/模型未在 PR 正文注明。

## 验证与回归风险

验证手段:benchmark 脚本内含正确性比对("All implementations match"),以及上表性能对比;PR checklist 的单测项未勾选,未新增单元测试。回归风险主要在 FP8 量化路径:新内核要求 hidden_dim % 16 == 0(原有 TORCH_CHECK 保留),`sm_count=132` 为硬编码,非 132-SM 卡(如 H100 114 SM、消费级卡)的切换阈值会偏离最优;warp 路径每 CTA 处理 8 token,对小 batch 仍回落 baseline,风险集中在很大 batch 时各 GPU 的表现。ROCm 分支保留了 `Float8_e4m3fnuz` 转换逻辑。

## 关联

同区域后续优化:PR #16725 在 per_token_quant_fp8 内核中引入 smem 支持,是同一内核的继续演进;其他 FP8 量化相关如 PR #12316(float8 转换加速)。本卡批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
