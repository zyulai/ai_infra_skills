# PR #18905: [jit kernel] Support per_token_group_quant_8bit jit kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-20 |
| 分类 | Quantization |
| Label | quant, sgl-kernel, run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/18905 |

## 一句话结论

新增 per-token-group 8bit 量化的 CUDA JIT kernel,大形状下较 Triton 实现最高快约 1.56x(768x16384:61.1→39.1 us)。

## 动机

per-token-group FP8/INT8 量化此前依赖 Triton 实现(`per_token_group_quant_8bit`),在大 token 数/大 hidden dim 场景有明显开销;sgl-kernel 侧已有 v2 实现,但作为预编译 wheel 交付,不易按输入 dtype 组合扩展。需要一个 JIT 版 CUDA kernel,按 (input dtype, output dtype) 组合即时编译,补齐非 v2 路径的性能与灵活性。

## 优化点

新 kernel `per_token_group_quant_8bit_kernel`:每个 128 等组由 16 线程处理,16 字节向量化加载,组内 absmax 用 warp shuffle 归约,避免跨 warp 同步;scale 支持 row-major float、column-major(TMA 对齐)与 UE8M0 打包(uint32/uint8)三种布局,UE8M0 走 `exp2f(ceilf(log2f(...)))` 的 2 的幂 scale;`compute_groups_per_block` 按 num_groups 整除性选 1~16 组/块平衡占用。Python 侧经 `load_jit` 按 (input.dtype, output_q.dtype) 缓存编译模块,`CPP_DTYPE_MAP` 补充 fp8_e4m3_t/int8_t;接入 `sglang_per_token_group_quant_fp8`:在 `enable_sgl_per_token_group_quant_8bit` 且非 `enable_v2` 时路由到 JIT kernel。注意该 JIT kernel 不支持 `fuse_silu_and_mul` 与 masked layout(基准里 silu+mul 是预计算后单独计时的),这些仍由 v2/sgl-kernel 或 Triton 承担。

## 关键改动

```text
python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit.cuh: 新增 CUDA JIT 量化 kernel
python/sglang/jit_kernel/per_token_group_quant_8bit.py: JIT 模块封装(cache_once)
python/sglang/srt/layers/quantization/fp8_kernel.py: 非 v2 分支路由到 JIT kernel
python/sglang/jit_kernel/tests/test_per_token_group_quant_8bit.py: 1872 个参数化用例
python/sglang/jit_kernel/benchmark/bench_per_token_group_quant_8bit.py: Triton vs JIT 基准
```

```cuda
local_absmax = GroupReduceMax(local_absmax, lane_id);
float y_s = local_absmax / max_8bit;
if constexpr (kScaleUE8M0) {
  y_s = exp2f(ceilf(log2f(math::max(y_s, 1e-10f))));
}
```

## 性能声明

kernel 基准(bench_kineto,fp8_e4m3fn,group_size 128,单位 us;GPU 型号 PR 未注明,测试在容器 c7e9bb6a6789):768 tokens x 16384 hidden:Triton 61.065 → JIT 39.095(约 1.56x);2048 x 7168:70.897 → 45.047;768 x 7168:27.809 → 18.694;256 x 16384:21.622 → 14.800;64 x 16384:6.856 → 5.435。小形状(1~16 tokens)JIT 反而略慢(如 1 x 1536:2.043 → 2.396)。MoE 多 rank 分支(hidden 2048,num_ranks 8~48,masked layout)在基准脚本中覆盖但正文未给出对比数字。PR 无 e2e 推理数字。

## 验证与回归风险

新增 `test_per_token_group_quant_8bit.py`:1872 个参数化用例(1~8192 tokens x 11 种 hidden x group 16/32/64/128 x 4 种 scale 布局 + MoE masked 变体),JIT 与 Triton 输出做 close/tiny-diff 对比;实测 1320 passed、552 skipped(4.46s)。风险面:仅在 `enable_sgl_per_token_group_quant_8bit` 且非 v2 时生效,默认路径不变;UE8M0 用例在架构 major<=9 时 skip(仅 Blackwell 需要);JIT kernel 不支持 fuse_silu_and_mul,依赖该特性的调用必须继续走 v2/Triton。

## 关联

同区域:sgl-kernel 侧 `sgl_per_token_group_quant_8bit`(enable_v2)为同函数族的预编译版本,本 PR 是其非 v2 路径的 JIT 替代;索引中另有 per-token quant smem 优化(如 pr-16725)。同作者 #13797 为 all-reduce 基准。具体前序 PR 编号素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
