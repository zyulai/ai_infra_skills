# PR #41713: Eliminate redundant MoE buffer copies in AITER fused experts (without dependency on AITER changes)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-mghanimi |
| 合并 | 2026-05-07 |
| 分类 | MoE |
| Label | performance, rocm, ready, deepseek |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/41713 |

## 一句话结论

ROCm AITER fused MoE 把调用方输出 tensor 直接别名给 fused_out,消除冗余 DMA copy:DSv3.2 TP4 每 decode step 少 116 个 copy kernel,per-step +2.3%。

## 动机

AITER fused MoE decode 的写回是 double-copy:kernel 先写 fused_out,`TopKWeightAndReduceNoOP.apply` 再 `output.copy_(result)`,每个 decode step 产生大量 `__amd_rocclr_copyBuffer` DMA kernel(DeepSeek-V3.2 TP4 为 116 个/step;MiniMax M2.5 场景 profile 达 12524 个事件),既浪费 launch 又挤占 custom allreduce 的显存带宽。

## 优化点

两处配合、不需要 AITER 侧改动:1) `modular_kernel.py` 的 `_fused_experts` 增加 `output_alias` 参数:在 ROCm 且 AITER fused MoE 启用时,若调用方 output 与 fused_out 的 shape/dtype/device 匹配且 contiguous,则直接 `fused_out = output_alias`,让 kernel 把结果写进调用方 buffer;2) `TopKWeightAndReduceNoOP.apply` 增加短路 `output is fused_expert_output` 时直接返回,免掉 self-copy;3) `rocm_aiter_fused_moe.py` 的 `output.copy_(result)` 改为守卫式 `output.set_(result)`:仅当 shape/dtype/device 匹配、双方 contiguous、output 是自身 storage 的根(`output._base is None`)且 `disable_inplace()` 允许时才 set_,否则保留 copy_。思路源自 #38597、ROCm/aiter commit `da318d0` 与 draft #41020。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/modular_kernel.py: output_alias 参数与别名守卫
vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py: self-copy 短路
vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py: copy_ → 条件 set_
```

```python
if current_platform.is_rocm():
    from vllm._aiter_ops import rocm_aiter_ops
    if (
        rocm_aiter_ops.is_fused_moe_enabled()
        and output_alias is not None
        and output_alias.shape == fused_out.shape
        and output_alias.dtype == fused_out.dtype
        and output_alias.device == fused_out.device
        and output_alias.is_contiguous()
    ):
        fused_out = output_alias
```

## 性能声明

- DeepSeek-V3.2 TP4(4x MI355X):每 decode step 移除 116 个 copy kernel,per-step 改善 2.3%(约省 500 µs);次要收益为 custom allreduce 的带宽争用减少。
- MiniMax M2.5 TP2 MI355X,ISL/OSL=1000/100,conc=64:`__amd_rocclr_copyBuffer` 12524 → 0 events;Memcpy DtoD 1307 → 807(−38%);output throughput 1512 → 1572 tok/s(+4.0%);median TPOT 33.6 → 32.2 ms(−4.2%)。
- torch profiler(ROCm 7.2.2,random 1000/100,conc 2,32 prompts):`__amd_rocclr_copyBuffer` 1.192 s / 198152 events → 0。
- 精度(lm_eval GSM8K):DSv3.2 TP4 hipGraph FULL_AND_PIECEWISE flexible 0.9310 / strict 0.9037,eager 0.9340/0.9083;MiniMax M2.5 TP2 hipGraph 0.9302/0.9257,eager 0.9264/0.9287。

## 验证与回归风险

验证:E2E 精度测试覆盖两个模型 × hipGraph/eager 两种模式 × 有/无 AITER MoE backend;torch profiler 事件计数确认 copy 消除。风险:`output.set_(result)` 会替换 output 的 storage,依赖 `output._base is None` 与 `disable_inplace()` 守卫规避视图/CUDA graph 场景,未覆盖的调用路径若假定 output 原 storage 稳定需注意;条件不满足时回退 `copy_`,行为与之前一致;别名仅 ROCm + AITER fused MoE 路径生效;`output is fused_expert_output` 短路影响所有走 `TopKWeightAndReduceNoOP` 的 modular MoE 路径(依赖上游正确传入 alias)。

## 关联

Draft 前序 #41020;同思路 #38597;ROCm/aiter commit `da318d0c`(modular_kernel 条件强化来源)。索引同区域另有 pr-56226(AITER topk_gating)、pr-28369(balance packing)、pr-46340(batched MoE operand loads)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
