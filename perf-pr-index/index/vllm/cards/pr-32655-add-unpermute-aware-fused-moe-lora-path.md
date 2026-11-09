# PR #32655: Add unpermute-aware fused MoE LoRA path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @RunkaiTao |
| 合并 | 2026-02-02 |
| 分类 | MoE (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32655 |

## 一句话结论

为 fused MoE LoRA 增加小 batch 下的 naive block assignment 路径,跳过 permute/unpermute 对齐 kernel,gpt-oss 120b e2e 基准时长缩短 15.2%、OTPS 提升 18%。

## 动机

fused MoE LoRA 原实现必须先跑 moe_lora_align_block_size 做 token/expert 的 block 对齐(permute),在小 batch、大专家数场景(max_lora=8 的 gpt-oss 120b 类模型)这一 kernel 及其 padding 开销占比很高,而激活的 experts*loras 组合只占很小一部分,对齐收益为负。

## 优化点

新增 naive_block_assignment fallback:当 `expert_map is None 且 num_tokens * top_k * SPARSITY_FACTOR(8) <= local_num_experts * max_loras` 时,跳过 moe_lora_align_block_size,直接令 expert_ids = topk_ids.view(-1)、sorted_token_ids = None、num_tokens_post_padded = None,按 token-wise 方式分配 block。Triton kernel 内把 lora_id/expert_id/token offset 的读取抽成 _get_lora_id/_get_expert_id/_get_token_offs 三个辅助 jit 函数,以 constexpr naive_block_assignment 分支:naive 模式下 lora_id 从 token_lora_mapping 按 token_idx = pid_m // top_k_num 取,expert_id 直接取展平的 topk_ids,token offset 用 `tl.where(offs == 0, pid_m, num_valid_tokens)` 实现每 program 单 token;grid 的 lora 维度从 max_loras+1 降为 1,去掉 -1 sentinel。token_mapping_factor 取代原 top_k constexpr,统一 shrink/expand 的 token 展开倍数。token_lora_mapping 全链路透传给 shrink/expand kernel。

## 关键改动

```text
vllm/lora/ops/triton_ops/fused_moe_lora_op.py: naive 分支的 _get_lora_id/_get_expert_id/_get_token_offs 与 kernel 签名改造
vllm/lora/layers/fused_moe.py: SPARSITY_FACTOR=8 触发条件,sorted_token_ids 可为 None,token_lora_mapping 传递
vllm/lora/punica_wrapper/punica_gpu.py: moe_lora_align_block_size 在 naive 模式下直接返回 topk_ids.view(-1)
tests/lora/test_fused_moe_lora_kernel.py: 新增 naive 路径参数化测试(1-8 tokens、64/128 experts)
benchmarks/kernels/benchmark_lora.py: shrink/expand kwargs 增加 token_lora_mapping
```

## 性能声明

E2E:gptoss 120b,input 1600/output 600,concurrency=8,tp=1,max_lora=8,num_request=200。baseline 输出吞吐 567.29 tok/s、Mean TPOT 13.01ms、Mean TTFT 667.10ms、时长 211.53s;本 PR 后 669.50 tok/s、11.00ms、578.74ms、179.20s。作者总结:benchmark duration 提速 15.2%,OTPS(output token throughput)提升 18%。GSM8K(before 0.603/invalid 0.098 → after 0.615/0.094)。kernel 级 shrink/expand 微基准数字在截图中,正文未给出具体数值,硬件未注明(E2E 与 kernel bench 均)。

## 验证与回归风险

单测:tests/lora/test_fused_moe_lora_kernel.py 的 naive 路径(num_tokens 1/2/4/8 x top_k 1/2 x experts 64/128 x max_loras 4/8 x rank 16/32),断言触发条件并对照 torch 参考;回归测试跑 gptoss_tp、fused_moe_lora_kernel、moe_lora_align_sum、deepseekv2_tp、qwen3moe_tp。回归面:仅影响 expert_map 为 None 且满足稀疏条件的调用(shape 不同走原路径);SPARSITY_FACTOR=8 是启发式阈值,大批量下不会命中;多 GPU(expert_map 非空)场景明确不走 naive 路径。

## 关联

无(正文与 diff 未提及相关 PR)。
