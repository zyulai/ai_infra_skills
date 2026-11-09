# PR #42290: [LoRA] Add one shot triton kernel For MoE LoRA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jeejeelee |
| 合并 | 2026-05-26 |
| 分类 | MoE(major) |
| Label | performance, intel-gpu, ready |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,8 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/42290 |

## 一句话结论

为 MoE LoRA 新增 shrink+expand 单次 launch 的 one-shot Triton kernel,消除中间缓存往返;另将双 stream 重叠扩展到 FusedMoEWithLoRA。

## 动机

此前 MoE 的 LoRA 计算走 `fused_moe_lora_shrink` + `fused_moe_lora_expand` 两个 kernel:先算 rank 维中间结果写入 intermediate cache,再由 expand kernel 读回并加权合并到输出。两次 launch 意味着中间张量要物化、多一次内存往返;小 batch/低 rank 场景下这部分开销占比明显。PR 正文极简(仅 "queued after #42757"),动机主要从 diff 注释还原。

## 优化点

新增 `_fused_moe_lora_one_shot_kernel`:把 shrink 与 expand 融合为一次 launch,rank 维 intermediate 直接留在寄存器中(BLOCK_R pad 到 next_pow2(rank)、下限 16 以满足 tensor core 要求),不再写 intermediate cache。调度上按 lora_id / expert_id 三维 grid(program_id.x 分 M×NPID_FACTOR,program_id.y 为 slice,program_id.z 为 lora),NPID_FACTOR 把 N 维切成多个外层块以在小 batch 时扩展并行度。`fully_sharded=False`(prefill 默认)走该 fast path;`fully_sharded=True` 保留旧两 kernel 路径,因为 shrink 与 expand 之间需要物化中间缓存做 all_reduce / all_gather。此外把 `_get_lora_aux_cuda_stream` 移入 `layers/utils.py` 并让 `FusedMoEWithLoRA` 也接入 `VLLM_LORA_ENABLE_DUAL_STREAM` 双 stream 重叠(base GEMM 与 LoRA fast path 并行,w13/w2 各用一对 event 同步)。

## 关键改动

```text
vllm/lora/ops/triton_ops/fused_moe_lora_op.py: 新增 _fused_moe_lora_one_shot_kernel 融合 shrink+expand
vllm/lora/layers/fused_moe.py: FusedMoEWithLoRA 初始化 aux stream 与 4 个 event,接入 MoELoRAContext
vllm/lora/layers/utils.py: _get_lora_aux_cuda_stream 移到公共位置并受 envs 门控
vllm/lora/layers/base_linear.py: 复用公共 stream helper
benchmarks/kernels/benchmark_fused_moe_lora_one_shot.py: one_shot vs two_kernel 对比基准
tests/lora/test_fused_moe_lora_kernel.py: one-shot 正确性与边界用例
```

关键片段(kernel 头注释):

```text
Fully-fused MoE-LoRA kernel (one-shot): shrink + expand combined into a
single launch with the rank-dim intermediate kept in registers. Used by
the fast path of `_fused_moe_lora` for `fully_sharded=False`. The legacy
two-kernel path is retained for `fully_sharded=True` because that path
needs to materialise the intermediate cache for an all_reduce /
all_gather between shrink and expand.
```

## 性能声明

PR 正文未给出量化 benchmark 数字(perf claims 检测为 none)。新增的基准脚本覆盖 mixtral / qwen3moe / glm5_1 三档预设(E=8/64/256),M ∈ {16, 64, 256, 1024, 4096, 16384},rank ∈ {8, 16, 32, 64},bf16,对比 one_shot 与 two_kernel;具体加速比 PR 未注明。

## 验证与回归风险

新增/扩展 `tests/lora/test_fused_moe_lora_kernel.py`:小 rank(4、8)下 rank 维掩码正确性、小 batch NPID>1 分支的跨外层块写掩码、三类早退条件(lora_id<0 / mapping 全 -1 / adapter 全禁用)下残差逐字节不变、num_active_loras=0 时不 launch 零维 grid、非法 block_size_m 报清晰断言。风险面:one-shot 与旧路径在 `fully_sharded` 分支间行为需一致;rank<16 的 pad 与 TMA 对齐在旧路径仍有已知限制(测试注释中说明);双 stream 路径依赖 event 配对正确性。

## 关联

前序依赖/排队于 PR #42757(正文注明 "queued after")。同区域:PR #34933(fused moe with LoRA shared expert dual stream)、PR #21525 同属 MoE/LoRA fast path 优化链。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
