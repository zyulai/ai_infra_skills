# PR #10321: [1/2][sgl-kernel] Support moe_sum_reduce cuda kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-09-19 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | 素材未提供 diffstat(diff 含 8 个文件,核心为新增 sgl-kernel/csrc/moe/moe_sum_reduce.cu) |
| 链接 | https://github.com/sgl-project/sglang/pull/10321 |

## 一句话结论

把 MoE 的 moe_sum_reduce 从 Triton 重写为 CUDA kernel:小 batch 快于 Triton,大 batch 持平。

## 动机

MoE 中的 `moe_sum_reduce`(对 topk 个专家输出按 token 求和并乘 routed_scaling_factor)此前是 Triton kernel(小 batch 时与 TorchCompile 版本配合)。作者将其重写为 CUDA kernel 以追求更好性能,目标是小 num_tokens 场景超越 TorchCompile 与 Triton,同时大 token_num 下不掉性能。

## 优化点

新增 `sgl-kernel/csrc/moe/moe_sum_reduce.cu`,按输入规模走三条路径:(1) bf16 向量化 fast path——`token_num > 256` 且 `hidden_dim % 8 == 0` 时启用 `moe_sum_reduce_warp_per_token_vec_kernel`,每 warp 负责一个 token,用 `uint4` 一次载 16 字节(`__ldg` + `__nv_bfloat16` 打包)并在 float32 累加;(2) `token_num > 128` 走 warp-per-token 的 `moe_sum_reduce_kernel_warp_token_topk`;(3) 小 token 走 block 型 `moe_sum_reduce_kernel`。TOPK 维度按 2/4/8/9/泛化做模板特化以展开循环。内核通过 `TORCH_LIBRARY` 注册为 `sgl_kernel::moe_sum_reduce`,Python 侧在 `sgl_kernel.moe` 暴露同名接口。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_sum_reduce.cu: 新增 CUDA kernel(三条路径 + topk 模板分发)
sgl-kernel/csrc/common_extension.cc: 注册 sgl_kernel::moe_sum_reduce op
sgl-kernel/python/sgl_kernel/moe.py: 暴露 moe_sum_reduce Python 接口
sgl-kernel/CMakeLists.txt / sgl-kernel/include/sgl_kernel_ops.h: 构建与声明
benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py: 加入 CudaKernel 对比曲线
```

内核分发逻辑(节选):

```cpp
const bool fast_bf16_vec_ok = (input.scalar_type() == at::kBFloat16)
    && (token_num > 256) && (hidden_dim % 8 == 0);
...
const bool per_token_use_one_warp = (token_num > 128);
```

## 性能声明

`benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`,单位 us(数值越低越好);PR 未注明 GPU 型号(提示符显示主机为 blackwell-research)与精度/输入形状,数值表未注明统计方式:

| num_tokens | Original | TorchCompile | Triton | CUDA |
| --- | --- | --- | --- | --- |
| 1 | 14.368 | 20.160 | 10.240 | 7.488 |
| 32 | 15.488 | 21.888 | 10.304 | 9.504 |
| 128 | 16.448 | 42.208 | 12.192 | 12.224 |
| 512 | 26.592 | 143.136 | 17.312 | 17.408 |
| 4096 | 114.336 | 1087.664 | 68.704 | 68.576 |

PR 结论:小 num_tokens 时 CudaKernel 同时优于 TorchCompile 与 TritonKernel;大 token_num 不掉性能。bs=4096 时 Triton 带宽近 3TB/s、约 75% 吞吐接近理论上限。正文将 75% 视为带宽利用率参照点而非本 PR 的提升数字。

## 验证与回归风险

benchmark 脚本内置正确性校验:`verify_correctness` 用 `torch.allclose(atol=1e-2, rtol=1e-2)` 对比 baseline/compiled/triton/cuda 四种实现,运行输出 "All implementations match"。PR 的 Accuracy Tests 一节为空,未提供模型级精度测试。回归风险:该 op 本卡范围内仅新增 sgl-kernel 侧能力,尚未见 sglang 主仓库调用切换(标题 [1/2] 表明为两步中的第一步);三条路径的阈值(256/128)与 hidden_dim 对齐约束决定实际生效路径。

## 关联

标题标注 [1/2],素材未给出配套第二步 PR 编号。同区域(MoE reduce/combine)其他 PR 编号素材未列出。
