# PR #2453: DeepseekMoE support with Fused MoE kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zwd003 |
| 合并 | 2024-01-30 |
| 分类 | MoE |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 部分可用,pack 注明另 dropped 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/2453 |

## 一句话结论

为 vLLM 引入 DeepseekMoE 模型与 Triton Fused MoE 内核,deepseek-moe-16b 吞吐较原路径最高 3.2x。

## 动机

vLLM 当时没有 DeepseekMoE(细粒度专家 + 共享专家)架构的支持:esmeetu 与 DeepSeek-AI 的早期实现(PR #2429)存在 bug 且未接入 fused MoE 内核,专家计算逐个走 PyTorch 循环。MoE 层每个 token 只激活 top-k 个专家,若为每个专家单独做 GEMM、且 token 分配未按块对齐,会产生大量小 kernel 与填充浪费。该 PR 修复 #2429 的 bug,补上专家并行(expert parallelism)与 fused MoE 支持。

## 优化点

机制上分三层。第一,新增 `moe_align_block_size` CUDA kernel:单 block 内用共享内存统计各专家 token 数、按 `ceil_div(count, block_size)*block_size` 计算前缀和,把 token 按 expert 排序并 pad 到 BLOCK_SIZE_M 的整数倍,产出 sorted_token_ids / expert_ids / num_tokens_post_pad,供 Triton 内核做块状矩阵乘(2024.01.21 该函数改用 C++ 实现后带来 10% 性能提升)。第二,新增 `fused_moe.py` 的 Triton kernel:一次遍历完成"按专家分块的 GEMM + routed weight 乘回",对 w1(w13)与 w2 两次调用,中间插入 silu_and_mul;小 batch 时切换到更小的 BLOCK 配置。第三,新增 Deepseek 模型实现,含 shared_experts 与 routed experts,`pack_params` 把专家权重展平成 (E, N, K) 的单一张量交给 fused MoE,inplace 求和写回 hidden_states。

## 关键改动

```text
csrc/moe_align_block_size_kernels.cu: 单 kernel 完成 token 按专家排序与块对齐 pad
vllm/model_executor/layers/fused_moe.py: Triton fused MoE kernel + align/pad + 两次 GEMM 调用封装
vllm/model_executor/models/deepseek.py: DeepseekForCausalLM 模型(共享专家 + 路由专家 + fused MoE)
tests/kernels/test_fused_moe.py: fused_moe 与 torch 参考实现的数值一致性测试
csrc/dispatch_utils.h: 新增 VLLM_DISPATCH_INTEGRAL_TYPES 宏
```

关键片段(moe_align_block_size 的前缀和对齐):

```cpp
cumsum[0] = 0;
for (int i = 1; i <= num_experts; ++i) {
    cumsum[i] = cumsum[i-1]
        + CEILDIV(tokens_cnts[blockDim.x][i - 1], block_size) * block_size;
}
*total_tokens_post_pad = cumsum[num_experts];
```

## 性能声明

2 x A100 40G PCIe,平均 prompt 长度 166 tokens,`benchmark_throughput.py` 类测试(prompt 数 32/256、max tokens 64/256、max batch 1/256):

- baseline vs 本 PR fused MoE(deepseek-16b-moe):256 prompts/64 tok/bs256:25s → 9s(2.8x);32 prompts/64 tok/bs1:104s → 33s(3.2x)。
- Llama2-13b vs deepseek-16b-moe(本 PR):256/256/256:51s → 23s(2.2x);32/256/1:396s → 140s(2.8x)。
- 对比 PR #2293 的实现:26s → 9s(2.9x)。
- C++ 版 align_block_size 后:较此前再 +10%,deepseek-moe-16b 速度接近 7B dense。
- 后续复测:`--input-len 1000 --output-len 64 -tp 2 --num-prompts 256` 下 deepseekmoe-16b 16587.82 tokens/s,超过 llama2-7b 的 10978.67 tokens/s。

## 验证与回归风险

新增 `tests/kernels/test_fused_moe.py`:对 m/n/k/experts/topk/dtype 参数网格(含 fp16/bf16)断言 fused_moe 与 torch 参考实现 allclose(atol=1e-2);正文另用 4 条 prompt 的生成输出做功能验证。风险:align kernel 假设 num_experts <= 64(NUM_MAX_EXPERTS,assert),更大专家数模型需后续扩展;fused MoE 仅支持 fp16/bf16;新 MoE 路径会覆盖所有后续采用 `fused_moe` 的模型,数值精度依赖 atol=1e-2 容差。

## 关联

前序:PR #2429(esmeetu/DeepSeek-AI 初版,本 PR 修 bug 并加 fused MoE)、PR #2293(对比基线)。作者提出的后续方向(另一 PR):融合 shared_experts 与 routed_experts 计算、融合 gate/softmax/topk。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
