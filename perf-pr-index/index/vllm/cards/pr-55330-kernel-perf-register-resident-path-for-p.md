# PR #55330: [Kernel][Perf] Register-resident path for per-token-group 8-bit quant

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chuan932 |
| 合并 | 2026-09-25 |
| 分类 | Quantization |
| Label | performance, quantization, verified |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/55330 |

## 一句话结论

给 per-token-group 8-bit 量化 kernel 增加 register-resident 调度,FP8 UE8M0 路径微基准最高 3.07x,端到端吞吐 +1.5~1.7%。

## 动机

`per_token_group_quant_8bit` 需要读 group 两次(一次求 absmax、一次量化),因此每组都要经 shared memory 暂存,且两次 pass 之间需 `__syncthreads()`。在 `group_size=128` 时一个 16-bit 输入 group 仅 256 B,完全可以放进寄存器,smem 暂存与同步开销均可省去。vLLM 已有同思路的 `per_token_group_quant_8bit_packed_register_kernel`,但它硬编码 UE8M0 指数按 4-per-int32 打包 + TMA 对齐 stride,float32 scale 的调用方无法使用。

## 优化点

新增 `per_token_group_quant_8bit_register_kernel`,面向普通(unpacked)float32 scale 布局:8 线程一组,每组 128 元素一次性以 `uint4` 载入寄存器,组内用 3 次 `__shfl_xor_sync` 完成 absmax 归约,量化后按 `uint4` 一次写回,全程无 shared memory 与 block 级同步。UE8M0 时 scale 为 2 的幂,内循环用乘倒数代替逐元素除法;非 UE8M0 路径保留除法以维持与 smem kernel 的 bit-exact。分发条件(`CanUseRegisterPath`):group_size==128、输入为 16-bit、输入/输出 buffer 均 16 B 对齐且 contiguous;其余一律走原 shared-memory 调度。固定 256 线程 launch 并对 grid 补 padding + 越界检查,消除了奇数 `num_groups` 时旧调度只剩 16 活跃线程的问题。

## 关键改动

```text
csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu: 新增 register kernel + CanUseRegisterPath 分发(约 150 行)
tests/kernels/quantization/test_per_token_group_quant.py: 新增 register vs smem bit-exact 守卫测试(30 case)
benchmarks/kernels/benchmark_per_token_group_quant.py: 补充 serving 规模 shape(1024/8192 x 6144/7168)
```

```cuda
inline bool CanUseRegisterPath(const torch::stable::Tensor& input,
                               const torch::stable::Tensor& output_q,
                               int64_t group_size) {
  const auto dtype = input.scalar_type();
  if (group_size != 128) return false;
  if (dtype != torch::headeronly::ScalarType::Half &&
      dtype != torch::headeronly::ScalarType::BFloat16) {
    return false;
  }
  if (!input.is_contiguous() || !output_q.is_contiguous()) return false;
  const auto in_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
  const auto out_addr = reinterpret_cast<uintptr_t>(output_q.data_ptr());
  return (in_addr % 16 == 0) && (out_addr % 16 == 0);
}
```

## 性能声明

微基准:H20-3e (SM90),CUDA 13.0,对 shipped shared-memory kernel,µs/call,输出 buffer 预分配。FP8 + UE8M0(DepGEMM 默认)、bf16 输入:(1,128) 14.0→5.5 (2.54x)、(1024,6144) 24.2→9.1 (2.66x)、(8192,6144) 183.9→59.9 (3.07x)、(65536,256) 62.9→22.1 (2.84x);fp16 (8192,6144) 3.25x。INT8 输出:1.32x~1.56x。非 UE8M0 FP8 约 2.1~2.3x。作者解释:~3x 是三个效应叠加(无 smem 往返、乘倒数、FP8 软件转换基线贵),register 侧在 (8192,6144) 已收敛到带宽受限。

E2E serving:GLM-5.2-FP8,TP=8 于 8xH20-3e,256k max model len,chunked prefill 8192,concurrency 1,`--kv-cache-dtype fp8`:16k~131k prompt 吞吐 +1.47%~+1.69%,8k +0.85%。profile 归因:8192-token chunk 下该 kernel 占 22.28 ms/step,按 3.03x/3.07x 估算约省 15 ms(~800 ms step 的 1.87%);测量经 Python 层 redirect(避免重编 `_C` so),每 call 多一层 Python 开销,故 1.5% 视为下界。

## 验证与回归风险

输出与 shared-memory kernel bit-exact(`torch.equal` 逐字节):独立扩展编译验证 FP8 72/72、INT8 10/10、经 `per_token_group_quant_fp8` 入口 48/48 配置 bit-exact,含 edge 输入(+inf/-inf/NaN/全零/1e-30/1e30)。新增测试:`test_per_token_group_quant_fp8_register_matches_smem`(24 case)与 int8 版(6 case),用 16B-misaligned 输入强制 smem 路径与 register 路径对比;`pytest -k register_matches_smem` 30 passed,原 fp8 套件 256 passed。风险面:ROCm shuffle 分支为复制已有模式但作者无 AMD 硬件实测;INT8 无端到端数字;非 group_size-128 / 32-bit 输入 / 未对齐 buffer 全部回退原调度,行为不变。

## 关联

同分类其他 PR(作者在 "Not a duplicate" 中列出):#53874、#54857、#43907(ROCm/aiter 将 quant 折入 GeGLU/RMSNorm/RoPE 的融合)、#53346、#51477(UE8M0 正确性修复)、#45151、#36297、#46935(折入其他 kernel epilogue)。索引内同区域卡:PR #18844(scaled FP8 quant 调优)、PR #32520(FP4 量化 kernel 优化)、PR #30885(smaller scaling 因子)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
