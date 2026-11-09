# PR #8499: [sgl-kernel performace] fix fp8 quant kernels dispatch __nv_fp8_e4m3 bug to improve performance 10%-20%

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2025-07-29 |
| 分类 | Quantization(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未记录);diff 可用,涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/8499 |

## 一句话结论

把 FP8 量化内核的输出类型从 `c10::Float8_e4m3fn` 软件转换改为 `__nv_fp8_e4m3` 硬件转换指令,量化 kernel 在 B200 上提速 10%-20%。

## 动机

FP8 per-tensor/per-token 量化是 FP8 推理的热点小 kernel,之前的实现把输出类型统一落在 `FP8_TYPE`(`c10::Float8_e4m3fn`)上。`c10::Float8_e4m3fn` 的 float->fp8 转换依赖软件实现而非硬件加速指令,导致指令数显著增加、分支预测开销上升、访存模式变化。作者在 ncu 的 SASS 里看到当前 kernel 为做转换反复调用子例程(连续多次 `CALL.REL.NOINC`),据此确认这是可修的性能 bug(承接 #8449 的结论)。

## 优化点

机制改动很小但精确:给三个量化 kernel 增加独立的 `DST_DTYPE` 模板参数,把原来写死的 `FP8_TYPE`(即 `c10::Float8_e4m3fn`)替换掉,host 侧 launch 时显式指定 `__nv_fp8_e4m3`。`__nv_fp8_e4m3` 来自 `cuda_fp8.h`,其 float->fp8 转换走硬件 FP8 转换指令,而 c10 类型走软件模拟,单条转换从多次子例程调用变成内联指令。量化数值逻辑本身(先 clamp 到 ±FP8_E4M3_MAX 再转换)不变;ROCm 分支(USE_ROCM 下仍用 `c10::Float8_e4m3fnuz` + `__hip_cvt_float_to_fp8`)保持原样未动。

## 关键改动

```text
sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu: per_tensor_quant_fp8_kernel 增加 DST_DTYPE 模板参数,launch 用 __nv_fp8_e4m3
sgl-kernel/csrc/gemm/per_token_quant_fp8.cu: per_token_quant_fp8_kernel 与 small_batch_kernel 同样模板化并改 launch 类型
sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu: 移除不再需要的 c10/util/Float8_e4m3fn.h 头文件
```

关键代码(launch 处,per_tensor_quant_fp8.cu):

```cpp
per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
    static_cast<scalar_t*>(input.data_ptr()),
    static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
    static_cast<float*>(output_s.data_ptr()),
    num_elements);
```

## 性能声明

条件:B200,`sgl-kernel/benchmark/bench_per_tensor_quant_fp8.py` 与 `bench_per_token_quant_fp8.py`,对比 main 分支与 PR 修复后,结论 "10%-20% performance improvement"。修复前 SGL Kernel per-tensor(batch 128, seq 2048)为 1868.095994,修复后 1525.455952;per-token(batch 128, seq 4096)修复前 3045.552015,修复后 2506.623983(表中未注明单位)。同时每行均显示 "All implementations match"(与 vLLM 参考实现数值一致),正文附有测试通过截图。

## 验证与回归风险

验证来自 benchmark 脚本内置的数值一致性检查(与 VLLM 参考实现对拍,全部 match),正文称 "Tests passed too" 并附截图;diff 未新增单元测试。回归风险低:改动只在非 ROCm 分支替换输出标量类型,数值路径(clamp 后转换)不变,ROCm 的 `Float8_e4m3fnuz` 路径完全未动;受益面是所有走 `sgl_per_tensor_quant_fp8`/`sgl_per_token_quant_fp8` 的 FP8 量化调用方(各类 FP8 模型的 activation quant)。

## 关联

前序:PR #8449(正文 "Follow" 链接),致谢 @strgb。同属 sgl-kernel FP8 量化 kernel 优化系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
