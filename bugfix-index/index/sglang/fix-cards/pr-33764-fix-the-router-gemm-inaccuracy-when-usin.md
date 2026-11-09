# PR #33764: Fix the router GEMM inaccuracy when using _front_w in Kimi-K3

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33764](https://github.com/sgl-project/sglang/pull/33764) |
| 作者 | b8zhong |
| 合并日期 | 2026-08-08 |
| labels | quant, run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Kimi-K3 的 `_front_w` 融合 GEMM 让路由 logits 变成 bf16 导致选专家不准,改为 GEMM 直接输出 fp32,其余消费者直接读 fp32。

## 问题现象
Kimi-K3 MoE 层用 bf16 计算路由 logits,router 基于这些 bf16 logits 选专家,选择不准确。用 layer-1 gate 权重与 fp32 correction bias 实测:5.27% 的 token 得到不同的 top-16 专家集合;平均 0.33% 的路由权重流向不同专家,p99 达 6.4%。作者指出这与 #29783 中 GLM-5.2 的问题同类(不安全,曾把 SWE-Bench Verified 从 79 掉到 67);且该路径以 num_tokens 为门控,大 batch 跑评测、小 batch 实际使用时表现不同,未来很难诊断。

## 根因
正文披露:`_merge_front_weights()` 把读同一 `hidden_states` 的三个投影(shared_experts.gate_up_proj `[1536,7168]`、router gate `[896,7168]`、routed_expert_down_proj `[3584,7168]` at TP8)沿行轴拼成 `_front_w [6016, 7168]`,用一个 GEMM 替代三个(每层省 11-16 us)。但 `fused` 是一个张量、一个张量只有一个 dtype:三个消费者中两个(SiTU、MXFP8 quant)需要 bf16,于是 router 也拿到了 bf16——功能上能跑,数值上不正确。其他实现(vLLM 的 gate、SGLang EP front `moe_front.fused_front`)对该 GEMM 均用 `out_dtype=torch.float32`。

## 修复方案
让 GEMM 直接写 fp32,另两个消费者直接读 fp32,不需要额外 cast 或新 kernel:

1. `python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py`:`cutedsl_bf16_gemm` 支持写 fp32 目标(`out_dtype` 可为 bf16 或 fp32;累加器本就是 fp32,只是去掉 epilogue cast,操作数仍 bf16),保持 TGV kernel 在 m<=8 比 cuBLAS 快 3us 的形态。
2. `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh` 与对应 Python 入口:`situ_and_mul` 模板拆为 `TIn/TOut`,支持输入输出不同类型(内部本就转 fp32 计算)。
3. `per_token_group_quant.cuh` 的 `QuantTrait` 增加 fp32 输入路径(`run_fp32`);`kVecSize` 固定为 16 个元素不再依赖输入宽度;`route_quant_fused.cuh` 按激活 dtype 模板化。
4. `kimi_k3.py`:新增 `_front_fp32`(非 HIP 且 `_front_w` 为 bf16 时融合 front 用 fp32 输出);`_k3_bf16_gemm` 支持 `out_dtype`;`_forward_fused` 中 `routed_input` 按需 `.to(hidden_states.dtype).contiguous()` 回 bf16(与原 bf16 front 位相同),其他 runner(marlin、triton)的 dense copy 顺带完成 cast。

实测:TP8 decode 各 token 数下每 MoE 层 front 耗时全面下降(如 128 tokens:34.82→33.57us);shared-expert 激活对 fp32 参考的最大相对误差从 3.9e-03 降到 2.5e-03(更准)。`routed_input` 位不变(`bf16(tgv_fp32)==tgv_bf16`)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 Kimi-K3 融合 front(`_front_w`)路径:CUDA 上 SM90 与 Blackwell 的所有 runner 路由改读 fp32 logits;`situ_and_mul`、`per_token_group_quant`、`route_quant_fused` 新增 fp32 输入分支;SM100 trtllm-gen mxfp4 之外的 runner 的 `routed_input` 多一次 cast(仅 1 token batch 多一次 launch)。ROCm 保持 bf16 front(`torch.mm` 的 `out_dtype` 与 aiter 路由未验证 fp32 链)。`_moe_front_needs_contiguous` 被 `_moe_front_needs_dense_bf16` 替代,门控条件变化需关注各 mxfp4 精度/SM 组合。另注:`test_per_token_group_quant_8bit_v2.py::test_v2_jit_matches_aot` 在 main 上本就失败 66 例,与本 PR 无关(作者已用移除改动的方式确认)。

## 关联
- https://github.com/sgl-project/sglang/pull/29783(GLM-5.2 同类精度问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
