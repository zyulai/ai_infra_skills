# PR #29929: [Kernels] Remove BatchedTritonOrDeepGemmExperts and default fallback to Triton

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bnellnm |
| 合并 | 2025-12-03 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, performance, frontend, ready, ci/build, v1, multi-modality, gpt-oss, kv-connector, nvidia |
| Diff | diffstat 缺失(pack 未记录 +N/-N);diff 内容可用(含删除 180 行的 batched_triton_or_deep_gemm_moe.py) |
| 链接 | https://github.com/vllm-project/vllm/pull/29929 |

## 一句话结论

删除 batched MoE 的 Triton/DeepGEMM 二选一包装层,改为显式选择:DeepGEMM 兼容且已安装就用,兼容但缺失则报错,否则回落 BatchedTritonExperts。

## 动机

`BatchedTritonOrDeepGemmExperts` 在 `BatchedTritonExperts` 与 `BatchedDeepGemmExperts` 之间做运行时二次分发的包装层:构造时按 `allow_deep_gemm` 决定内部持有哪个实现,`apply` 再转发,`workspace_shapes` 要按两个实现取较大者。这层间接让"请求了 DeepGEMM 但没安装"或"层不兼容"的失败被静默回落 Triton,选择逻辑分散、难以排查。

## 优化点

整文件删除 `batched_triton_or_deep_gemm_moe.py`(180 行)。在 `compressed_tensors_moe.py` 的 `select_gemm_impl` 改为显式决策:batched 激活格式下,`use_deep_gemm = envs.VLLM_USE_DEEP_GEMM and envs.VLLM_MOE_USE_DEEP_GEMM`;若开启且层兼容 DeepGEMM(`use_fp8_w8a8` 且 `block_shape == get_mk_alignment_for_contiguous_layout()`,即 fp8 + 128 block 量化)且 `has_deep_gemm()` 为真,返回 `BatchedDeepGemmExperts`;环境变量要求 DeepGEMM 但包未安装,或层兼容而包未安装时,抛 `RuntimeError`;否则返回 `BatchedTritonExperts`。standard 激活格式仍走 `TritonOrDeepGemmExperts`(传 `allow_deep_gemm=use_deep_gemm`)。这是选择策略/代码结构清理,kernel 本身不变;性能含义是"能走 DeepGEMM 就一定走,走不了就显式失败",避免静默落到慢路径。同步更新 docs 特性表与测试工具注册。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/batched_triton_or_deep_gemm_moe.py: 整文件删除
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py: select_gemm_impl 显式选择,DeepGEMM 缺失抛 RuntimeError
docs/design/moe_kernel_features.md: 删除 deep gemm+triton 行,更新 deepep_low_latency/pplx 家族表
tests/kernels/moe/modular_kernel_tools/mk_objects.py、vllm/model_executor/layers/fused_moe/__init__.py: 移除注册与导出
```

## 性能声明

PR 正文无 benchmark 数字(Test Plan 仅写 CI)。作者未给出前后吞吐对比;本 PR 的价值在于 MoE backend 选择路径的确定化与可诊断性。

## 验证与回归风险

测试计划为 CI(含 modular kernel 测试工具中删除的注册项)。主要语义变化:环境变量开启 DeepGEMM 但未安装时,行为从"静默回落 Triton"变为启动报错,这是刻意的。影响面为 batched 激活格式(DeepEP low latency / pplx 等路径)的 compressed-tensors fp8 block 量化 MoE;standard 格式路径不变;`workspace_shapes` 不再需要按两个实现取大。

## 关联

同区域保留:standard 格式的 `TritonOrDeepGemmExperts`、`BatchedDeepGemmExperts`、`BatchedTritonExperts`。素材未给出关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
