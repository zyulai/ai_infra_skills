# PR #6919: [sgl-kernel] Add cuda kernel for moe_ep_silu_and_mul

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-06-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | high priority, ready-to-merge |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/6919 |

## 一句话结论
把 EP MoE 的 silu_and_mul 从 Triton 重写为 CUDA kernel,作者报告约 10% 性能提升。

## 动机
EP MoE 流水线中的 `moe_ep_silu_and_mul` 此前由 Triton kernel(`silu_and_mul_triton_kernel`)实现,作者将其用 CUDA 重写以获得约 10% 性能提升。PR 正文未提供更细的瓶颈分析或 profile 数据。

## 优化点
新增 CUDA kernel `ep_moe_act_and_mul_cuda_kernel`:每个 token 一个 block,按 `reorder_topk_ids` 查专家 id,不在 `[start_expert_id, end_expert_id]` 范围内直接返回;对 gate/up 两半 hidden 做 16 字节向量化访存(借 flashinfer `vec_t`,vec_size = 16/sizeof(scalar_t)),在寄存器内完成 SiLU、与 up 相乘、乘 expert scale(scales 取倒数)后写出;尾部不足一个向量的部分用标量循环兜底。模板化支持 fp16/bf16/fp32;block 线程数按 half_hidden/vec 计算并对齐到 32,上限 1024。kernel 经 `TORCH_LIBRARY` 注册为 `sgl_kernel.ep_moe_silu_and_mul` 并暴露 Python 绑定。

## 关键改动

```text
sgl-kernel/csrc/moe/ep_moe_silu_and_mul_kernel.cu: 新 CUDA kernel(向量化 SiLU*up*scale,模板化 fp16/bf16/fp32)
sgl-kernel/csrc/common_extension.cc / include/sgl_kernel_ops.h / python/sgl_kernel/moe.py: op 注册与 Python 绑定
sgl-kernel/benchmark/bench_moe_silu_and_mul.py: batch 64–4096 × hidden 1024–8192 的 CUDA vs Triton 对比 benchmark
sgl-kernel/tests/test_ep_moe_silu_and_mul_kernel.py: CUDA 与 Triton 输出一致性测试
```

## 性能声明
原文:"Rewrite moe_ep_silu_and_mul Triton kernel in CUDA. Gains 10% performance improvement."。PR 未注明测试条件(模型/硬件/workload 均未给出)。

## 验证与回归风险
新增 pytest 用例在 total_tokens {32,256,1024} × hidden {128,256,512} × dtype {fp16,bf16,fp32} 组合下对比 CUDA kernel 与原 Triton kernel 输出(rtol=atol=1e-5)。风险:该 kernel 服务于 EP MoE 的激活路径,语义包含对 expert scale 取倒数(1/scales)的处理,需与下游 EP 代码对 scale 的约定保持一致;CMakeLists 新增源文件,需要重编 sgl-kernel。

## 关联
无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
