# PR #51674: [Kernel][Perf] Add fused CUDA post-conv MTP decode kernel for Qwen3.5 GDN

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Jie-Fang |
| 合并 | 2026-08-14 |
| 分类 | Speculative Decoding |
| Label | ready, ci/build, qwen, nvidia |
| Diff | diffstat 缺失;pack 明示 diff 因超长截断(8 个文件未含) |
| 链接 | https://github.com/vllm-project/vllm/pull/51674 |

## 一句话结论

Qwen3.5 GDN MTP decode 全步融合为单个 CUDA kernel,B200 cold-cache 快 1.17–2.20x。

## 动机

Blackwell 上 Qwen3.5(GDN linear attention)MTP speculative decode 时,Triton 路径每步要发起一串小 kernel(gating、delta-rule recurrence、state rewind/update、gated RMSNorm),在 decode batch size 区间让 GPU 处于 latency-bound 状态。

## 优化点

新增单个融合 CUDA kernel `fused_gdn_decode_post_conv_mtp`:消费 post-convolution 的 `mixed_qkv`,一次 launch 完成整个 GDN MTP decode step——q/k/v 的 SiLU 激活与 gating(`a`/`b`/`A_log`/`dt_bias` 的 softplus/sigmoid)、每请求最多 8 个 speculative token 的 gated delta-rule recurrence(按 `num_accepted_tokens` rewind 到最后被接受的 token)、in-place recurrent state 更新(BF16 或 FP32 state)、gated RMSNorm epilogue(输出精度对齐 Triton 路径)。实现上 grid 为 `(num_requests, num_value_heads)`、每 block 256 线程,state 用 `cp.async` 双缓冲按 32 行 chunk 流水加载,q/k 归一化与 warp 归约均在共享内存/寄存器完成。启用条件严格:要求 `num_v_heads == 8 * num_k_heads` 的 GDN head 布局且 SM100+;不满足时 dispatch 检查不会选中融合 kernel,默认行为不变(现有公开 Qwen3.5 checkpoint 0.8B/35B/397B 均非该布局)。op 注册由 `VLLM_ENABLE_FUSED_GDN_DECODE` 编译开关控制。

## 关键改动

```text
csrc/libtorch_stable/gdn/fused_gdn_decode_post_conv_mtp kernel(新文件,557 行): 融合 gating+recurrence+rewind+state 更新+gated RMSNorm
csrc/libtorch_stable/torch_bindings.cpp / ops.h: _C.fused_gdn_decode_post_conv_mtp 注册(VLLM_ENABLE_FUSED_GDN_DECODE)
tests/kernels/mamba/test_gdn_fused_mtp.py: model-path dispatch 测试(pure/mixed/prefill/decode)
tests/kernels/test_fused_gdn_post_conv.py: kernel 直测(BF16/FP32 state、ragged acceptance、tp4/tp16)
```

## 性能声明

1x B200(SM100)、CUDA 13.1、torch 2.13.0+cu130。对比边界相同(post-conv `mixed_qkv` 进,gated+normed `core_attn_out` 出,state in-place 更新);CUDA graph 捕获 20 次迭代、replay 50x 计时(生产 MTP decode 跑在全图下,故取 graph-replay GPU time,双方均不含 CPU launch 开销);cold 轮换 buffer 使 working set 超 4x L2(126 MB),warm 复用同一 buffer;K=V=128、4 MTP tokens/request、`num_accepted_tokens=2`,结果与 seqlen 无关。**Cold-cache speedup 1.17x–2.20x**,在 latency 敏感的 BS=4–32 最大:TP4 bf16 state BS=4 为 27.91→12.99 µs(2.15x),TP4 fp32 BS=4 2.20x,TP16 bf16 BS=16 2.13x;BS=128 仍 1.17–1.49x。lm_eval 未跑:无匹配 head 布局的公开 checkpoint。

## 验证与回归风险

测试结果(1x B200):`test_gdn_fused_mtp.py` 6 passed、`test_fused_gdn_post_conv.py` 70 passed、`test_gdn_metadata_builder.py` 10 passed、envs 1 passed、compile config 44 passed。kernel 输出与更新后 state 对照 `fused_sigmoid_gating_delta_rule_update` + `rmsnorm_fn`(atol=rtol=3e-2,非逐位),dispatch 测试确认 mixed/prefill/decode batch 正确回退 Triton 路径、融合调用次数符合预期。风险:仅 SM100+ 生效;HV/H=8 布局门槛使现役模型默认不受影响;数值等价为容差级;pack 明示 diff 截断(8 个文件未含,含 Python dispatch 侧)。

## 关联

无(正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
