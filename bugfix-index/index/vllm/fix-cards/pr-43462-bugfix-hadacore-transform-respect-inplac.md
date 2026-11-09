# PR #43462: [Bugfix] hadacore_transform: respect inplace parameter to fix garbage outputs with QuIP transforms

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43462](https://github.com/vllm-project/vllm/pull/43462) |
| 作者 | Yatimai |
| 合并日期 | 2026-09-22 |
| labels | bug, nvidia, quantization, ready |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

hadacore 内核忽略 inplace,mutable schema 又让 inductor 消除保护 clone,QuIP 在 compile+cudagraph 下输出乱码;改写 out 指针并传 inplace=False。

## 问题现象

使用 QuIP(在线 Hadamard 变换)的 compressed-tensors checkpoint 在 vLLM 默认 torch.compile + cudagraph 下输出乱码:RTX 5090 (SM120) 上 W4A16+QuIP、NVFP4A16+QuIP 修复前均乱码;B200 (SM100) 用 v0.21.0 预编译 wheel + torch 2.11.0 复现,WikiText-2 PPL ~1.4e6,而 `enforce_eager=True` 时正常(6.84)。main 构建 @4dcd10eb0 上 5 个 sanity prompt 全部乱码,dynamo 同时发出输出别名输入的 UserWarning。#28748 的 gsm8k_eval (0.697) 使用的组合不触发 inductor 消除;是否触发取决于 (torch 版本 × compile/cudagraph) 配置而非硬件。还阻塞 MXFP4 W4A4 + MR-GPTQ 等依赖在线 Hadamard 旋转的路径。

## 根因

正文披露两个相互作用的问题:(1) 潜在内核 bug——`hadacore_transform` 声明为 mutable schema(`Tensor! x`),但 CUDA kernel 无论 `inplace` 取值都写 `x.data_ptr()`,`inplace=False` 调用方拿到的 `out` 从未被写入;(2) `CompressedTensorsLinearTransformMethod.apply()` 中 #26849 引入的 `.clone()` 被 inductor 基于 `Tensor!` schema 判为冗余并从 FX 图中消除(经 `TORCH_LOGS="output_code"` 验证),kernel 收到别名 `x` 的切片 view,在 compile+cudagraph 下产生乱码。

## 修复方案

`csrc/libtorch_stable/quantization/hadamard/hadacore/hadamard_transform_cuda.cu`:`run_fht` 改为写 `out.data_ptr()` 而非 `x.data_ptr()`(尊重 inplace;inplace=True 时 out 即 x);`vllm/model_executor/layers/quantization/compressed_tensors/transform/module.py` 两处调用改为 `ops.hadacore_transform(value, inplace=False)` 选择非别名路径;保留 #26849 的 `.clone()` 作为防御层。

```diff
-      hadacore::run_fht<SCALAR_TYPE>(x.data_ptr(), x.data_ptr(), x.numel(), had_size, stream);
+      hadacore::run_fht<SCALAR_TYPE>(x.data_ptr(), out.data_ptr(), x.numel(), had_size, stream);
```

## 影响与回归风险

- 影响窗口:正文在 v0.21.0、v0.22.1(g0decac0d9)与 main @4dcd10eb0 上复现;触发取决于 (torch 版本 × compile/cudagraph) 组合。
- 回归面:hadacore CUDA kernel 的输出指针(影响所有 `hadacore_transform` 调用,inplace=True 语义不变)与 compressed_tensors 变换的 head_dim/sylvester 两条 fast path(现显式 inplace=False)。正文的无回归检查:NVFP4A16 无 QuIP 输出 byte-identical、kernel inplace 两模式 max diff 0.0、gsm8k 0.697 持平。

## 关联

vllm-project/llm-compressor#2006、#26849、#28748、#28747、#22486

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
