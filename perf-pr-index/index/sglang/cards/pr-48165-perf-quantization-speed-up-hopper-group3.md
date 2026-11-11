# PR #48165: [Perf] quantization: Speed up Hopper group32 FP8 quantization with FP32 scales

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Frank-whw |
| 合并 | 2026-11-09 |
| 分类 | Quantization(tier: major) |
| Label | quant, run-ci, jit-kernel |
| Diff | +332/-111,12 files(pack diffstat 缺失,按 pack 内 diff 文本计数) |
| 链接 | https://github.com/sgl-project/sglang/pull/48165 |

## 一句话结论

Hopper 上 BF16 group32 的 FP32 pow2 scale 量化改走 trait kernel,冷 L2 3.16×,8K TTFT 降约 2.2%。

## 动机

row-major FP32 power-of-two scale 路径此前一律走 `per_token_group_quant_8bit_v2`。与之匹配的 contiguous BF16 group32 case 在 Hopper 上可以复用已有 trait-driven kernel:baseline 每 CTA 32 线程/16 组,trait kernel 256 线程/128 组,[8192,5120] 的 grid 从 81,920 降到 10,240 CTAs,并用 packed BF16 pairs 计算 local absmax 与 power-of-two scaling。基于 #26766 的 fused rounding;#46838 处理 column-major FP32 scales(DeepGEMM),本 PR 面向 row-major group32。

## 优化点

把"scale 舍入到 2 的幂"(`kUe8m0`)与"packed 存储"(`kPackedScale`)拆成独立模板参数:FP32 scale 情形把 scale_inv 存为 `__uint_as_float(exp << 23)` 而非 uint8 字节。Python dispatch 重写:满足 2D BF16、contiguous FP32 scales、group32、eps=1e-10、无 silu/masked 的流量在 SM90 且输入 32B 对齐时走 trait kernel;16B 对齐 view、其他架构及其他 FP32 pow2 组合保留 v2 fallback。device capability 在 opaque custom op 内运行时查询,去掉 import 期 CUDA/NVML 探测,保持 Dynamo fullgraph 可追踪。BF16 UE8M0 简化为 `__hmul2` 加饱和 FP8 转换(移除 v2 的 NaN→−448 特判;FP16 因 multiplier 可能溢出保留 sanitization,并以编译期 static_assert 强制 FP16+FP32 pow2 走 v2)。JIT v2 API 移除 eps/min_8bit/max_8bit 参数(固定 absmax floor 1e-10 与 dtype 派生 range)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh: QuantTrait 增加 kPackedScale,FP32 power-of-two scale 存储与 BF16 简化
python/sglang/kernels/ops/quantization/fp8_kernel.py: FP32 pow2 dispatch 重写,收敛到 trait kernel 的条件
python/sglang/kernels/ops/quantization/per_token_group_quant.py: custom op 内运行时 SM90/32B 对齐 gate 与 singleton stride 归一化
python/sglang/kernels/jit/csrc/gemm/per_token_group_quant_8bit_v2.cuh: 为 fallback 增加 2D BF16 group32 buffer 校验
test/registered/kernels/ops/quantization/test_per_token_group_quant.py 等: 新增 test_fp32_pow2_* 用例与 benchmark_fp32_pow2
```

```cuda
const auto exp = cast_to_ue8m0(raw_scale);
if constexpr (kPackedScale) {
  scale_inv = static_cast<uint8_t>(exp);
} else {
  scale_inv = __uint_as_float(static_cast<uint32_t>(exp) << 23);
}
```

## 性能声明

4×H200,DeepSeek-V4.1-Flash-1017,TP4/EP4,checkpoint FP8 dense 层,Humming MXFP4 experts+FP8 activations,无投机;6 次交替 A/B server 重启,每值取三轮中位数再跨重启取中位。P50 TTFT:4K/1 c1 113.669→111.083 ms(-2.27%)、8K/1 c1 -2.21%、8K/1 c2 -2.31%;文档 4111/32 c2 -2.86%;128/128 TPOT 6.887→6.842 ms(-0.65%)。冷 L2 microbench [8192,5120] 100.257→31.761 μs(3.16×);[1,1536] 回归 +0.103 μs(如实保留,动机是 prefill)。kernel profile:baseline 100.29-100.42 μs vs candidate 29.79-30.22 μs @ [8192,5120]。GSM8K 前 200 题六轮交替 195-198/200(base)vs 196-197/200(candidate);对同激活回放 v2 与 candidate 共 65,036 次采样调用零比特不匹配。

## 验证与回归风险

新增 test_fp32_pow2_* 系列:全 finite BF16 编码(均匀/混合组)、scale 边界、singleton scales、16B-offset view、CUDA graph replay、torch.compile(fullgraph,eager/Inductor)、FP16/FP32/INT8 fallback;最终 H200 跑 382 tests(1 skipped)。风险:共享 C++ 模板改动,Blackwell/ROCm 无本地执行(靠 CI 验证);FP16+FP32 pow2 强制走 v2;非 32B 对齐输入走 v2;[1,1536] 小形状有 0.103 μs 回归。

## 关联

基于 #26766(fused rounding);相关 #46838(column-major FP32 scales / DeepGEMM)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
