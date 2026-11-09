# PR #57092: [Bugfix] Make online FP8 MoE experts batch-invariant under VLLM_BATCH_INVARIANT

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ybwbqg9379 |
| 合并 | 2026-10-23 |
| 分类 | MoE(tier: major) |
| Label | bug, ready, quantization |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/57092 |

## 一句话结论

修复 `VLLM_BATCH_INVARIANT=1` 下 online FP8 MoE 专家输出随 batch 组成变化的问题,使 BS=1 与 BS=8 的 logprobs bitwise 一致。

## 动机

Fixes #57016:`--quantization fp8`(online per-tensor FP8,Triton experts)的 MoE 模型在 batch-invariant 模式下,greedy token 在前约 10 个位置内翻转,prefix-cached K/V 也随写入者不同而不同;dense 层已不变,只有 experts 不变。根因之一是专家输入用覆盖整个激活 tensor 的单一动态 scale(`kFp8DynamicTensorSym`)量化——scale 是 batch 内所有请求所有 token 的 max,输出自然依赖 batch 组成。

## 优化点

两处修复:(1) 量化粒度切换——`Fp8PerTensorOnlineMoEMethod` 与 FP8 序列化 per-tensor checkpoint + dynamic activation 的 `Fp8MoEMethod`,在 `VLLM_BATCH_INVARIANT` 下改选 `kFp8DynamicTokenSym` 并传 `per_act_token_quant=True`,镜像 dense 层 `Fp8LinearMethod` 的做法;per-token 量化严格细于 per-tensor。(2) 顺带暴露并修复 launcher 潜在 bug——`invoke_fused_moe_triton_kernel` 对 1-D `[E]` weight scale 传 `stride_bse=0`(kernel 的 per_channel_quant 分支会拿 expert 0 的 scale 顶替所有 expert,或越界读),对 `[E,1,1]` 的单例维传 `stride_bsn=1`;修复为无条件传 expert stride 并对单例 scale 维广播(`B_scale.size(1) > 1` 才取 stride),`fused_batched_moe.py` 同样处理。组合测试的 NYI gate 相应收窄为"per-channel weights + per-tensor activations",使"per-tensor weights + per-token activations"组合进入覆盖。bf16 Triton expert 路径在 main 上本已不变(tile config 在该 flag 下被 pin)。

## 关键改动

```text
vllm/model_executor/layers/quantization/online/fp8.py: batch-invariant 下 activation_key 选 kFp8DynamicTokenSym,设 per_act_token_quant
vllm/model_executor/layers/quantization/fp8.py: Fp8MoEMethod 同样切换 + get_fused_moe_quant_config 传递 per_act_token_quant
vllm/model_executor/layers/fused_moe/fused_moe.py: invoke_fused_moe_triton_kernel 修复 [E] scale 的 stride_bse 与单例维 stride_bsn
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py: invoke_moe_batched_triton_kernel 同样广播单例 scale 维
```

## 性能声明

数字均为正确性/精度验证(RTX 5090 sm_120 WSL2,torch 2.13.0+cu130;另由 @CarrotSwordsman 在 H20 sm_90 独立复验):

- Kernel 层(`fused_experts` vs `torch_experts`,E=32, N=1024, K=2048, M=128, topk=4,per-tensor `[E,1,1]` weight scales):max abs err per-tensor 激活 0.010、per-token 激活 0.003(launcher 修复前 per-token 场景返回数千量级的值)。
- gsm8k 5-shot,1319 题,granite-3.1-1b-a400m-instruct `--enforce-eager`:默认 34.1%,`VLLM_BATCH_INVARIANT=1` on main 34.3%,`VLLM_BATCH_INVARIANT=1` + 本 PR 32.7%;PR 归因于 1B 模型的采样噪声。
- 复现 probe(before → after):A-alone vs A-with-B 首 token 在位置 11 翻转(logprob gap 0.166)、warm-vs-cold 翻转(gap 0.347)→ 修复后全部对比逐位一致。

## 验证与回归风险

新增 `tests/v1/determinism/test_batch_invariance.py::test_online_fp8_moe_logprobs_bitwise_bs1_vs_bsN`(granite-3.1-1b-a400m-instruct,`fp8_per_tensor`,greedy,BS=1 vs BS=8 bitwise tokens+logprobs;main 上 FAIL,diff 2.2e-01;PR 上 PASS);`test_batched_mm_per_tensor_weight_per_token_act` 新增 `[E]`/`[E,1,1]` 两种 scale 形状;`test_modular_kernel_combinations.py` 移除 per-tensor-weight + per-token-act 的跳过。回归风险:batch-invariant 模式下 FP8 MoE 激活量化从 per-tensor 变为 per-token,计算量与 scale 存储增加(1B 模型 gsm8k -1.4pt 属可接受范围);`B_scale` stride 逻辑变化影响所有走 `invoke_fused_moe_triton_kernel` 的 per-channel 分支,需要 block-fp8 套件守护(164 passed/20 skipped,test_fused_moe 241 passed)。

## 关联

修复 #57016;#55881 为 XPU MoE 变体(不碰 CUDA Triton 路径);#53585 之后 online per-tensor FP8 的选择命令为 `--quantization fp8_per_tensor`。同区域可参考 batch-invariant / determinism 相关 PR 与 FP8 MoE 量化卡。
