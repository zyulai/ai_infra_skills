# PR #8499: [sgl-kernel performace] fix fp8 quant kernels dispatch __nv_fp8_e4m3 bug to improve performance 10%-20%

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8499](https://github.com/sgl-project/sglang/pull/8499) |
| 作者 | BBuf |
| 合并日期 | 2025-07-29 |
| labels | (none) |
| 热度 | 👍 12 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论

FP8 量化内核误用软件实现的 `c10::Float8_e4m3fn` 做转换,改为硬件 FP8 类型 `__nv_fp8_e4m3`,B200 上提速 10%–20%。

## 问题现象

sgl-kernel 的 per-tensor / per-token FP8 量化内核转换浮点到 FP8 时,实际走的是 `c10::Float8_e4m3fn` 的软件实现转换路径。作者在 NCU 的 SASS 代码里观察到内核用大量指令做 float 到 fp8 的转换(9 次以上的 `CALL.REL.NOINC` 软件转换调用),性能明显低于 VLLM 参考实现:main 分支 B200 上 per-tensor 量化基准中 SGL Kernel 落后 VLLM 约 20%–45%(如 128x2048:1868 vs 2327 GB/s),per-token 动态量化同样全面落后。

## 根因

按 PR 正文:性能劣化的根源是 `c10::Float8_e4m3fn` 依赖软件实现的转换函数,而不是硬件加速的 FP8 转换指令。这导致指令数显著增多、分支预测开销上升、内存访问模式改变。即 FP8 类型 `FP8_TYPE` 被分发到 `c10::Float8_e4m3fn` 而非 CUDA 硬件类型 `__nv_fp8_e4m3`(标题所称的 dispatch bug)。

## 修复方案

给 `per_tensor_quant_fp8_kernel`、`per_token_quant_fp8_kernel`、`per_token_quant_fp8_small_batch_kernel` 增加独立模板参数 `DST_DTYPE`,宿主端 launch 时显式传入 `__nv_fp8_e4m3` 并将输出指针按其转型;移除 `per_token_group_quant_8bit.cu` 中的 `c10/util/Float8_e4m3fn.h` include。ROCm 分支的 `c10::Float8_e4m3fnuz` 软件转换保持不变:

```diff
-template <typename T>
+template <typename T, typename DST_DTYPE>
 __global__ void per_tensor_quant_fp8_kernel(
...
-    per_tensor_quant_fp8_kernel<scalar_t><<<grid, block, 0, stream>>>(
+    per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
```

B200 基准显示:per-tensor 量化 SGL Kernel 由 1868 提升到 2327 GB/s(128x2048,已追平 VLLM),per-token 由 3046 提升到 4045 GB/s(128x4096);整体提速 10%–20%,基准中 "All implementations match" 校验与测试均通过。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;影响使用 sgl-kernel FP8 量化内核的 CUDA 推理。
- 回归面:仅替换非 ROCm 路径的输出类型与转换指令,数值语义仍是 clamp 到 ±FP8_E4M3_MAX 后转换;`__nv_fp8_e4m3` 与原 c10 类型的舍入行为在正文未做逐位对比,但基准校验 "All implementations match" 通过。ROCm(`USE_ROCM`)路径不受影响。

## 关联

- #8449(本 PR 跟进的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
