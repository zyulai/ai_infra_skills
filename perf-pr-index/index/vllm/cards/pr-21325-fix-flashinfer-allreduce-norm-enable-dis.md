# PR #21325: Fix Flashinfer Allreduce+Norm enable disable calculation based on `fi_allreduce_fusion_max_token_num`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xinli-git |
| 合并 | 2025-07-22 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/21325 |

## 一句话结论

修复 allreduce+RMSNorm fusion 的启用判断 bug,使 FlashInfer fused kernel 在大 hidden size 下真正被调用。

## 动机

cudagraph warmup 传入的 all_reduce_in 形状是 [num_tokens, hidden_size](DSR1 的 hidden size 为 7168),但启用判断把 token 上限误写成 max_token_num * shape[0](即 max_token_num * num_tokens),两个量纲不匹配;hidden size 足够大时乘积恒超限,FlashInfer fusion 被无条件禁用,即使显式开启 enable_fi_allreduce_fusion 也从不生效。

## 优化点

collective_fusion.py 的 call_trtllm_fused_allreduce_norm 重写判断:current_tensor_size = num_tokens * hidden_size * element_size,与 max_fusion_size = max_token_num * hidden_size * element_size 比较,量纲一致;世界大小不在 _FI_MAX_SIZES 表中时回退到更保守的 _DEFAULT_FI_MAX_SIZE = MiB//2(512KB),而不是 KeyError。

## 关键改动

```text
vllm/compilation/collective_fusion.py: 修正 use_flashinfer 判定 + 新增 _DEFAULT_FI_MAX_SIZE 兜底
```

```python
num_tokens, hidden_size = allreduce_in.shape
element_size = allreduce_in.element_size()
current_tensor_size = num_tokens * hidden_size * element_size
max_fusion_size = max_token_num * hidden_size * element_size
use_flashinfer = current_tensor_size <= min(
    _FI_MAX_SIZES.get(world_size, _DEFAULT_FI_MAX_SIZE), max_fusion_size)
```

## 性能声明

DSR1 + 4xB200(hf_overrides 30 层),TP4,benchmark_serving random 1024 in/128 out(PR 自标 WIP benchmark)。profiler 证实修复前无 FI kernel(NCCL AllReduce 250.7ms/23.71%),修复后 flashinfer::trtllm_allreduce_fusion 被调用(407.5ms/31.55%)。TPOT/TTFT 对比(fusion vs no fusion,ms):conc1 TPOT 10.37 vs 10.61;conc8 13.03 vs 13.23;conc32 17.43 vs 17.50;conc512 144.32 vs 144.74。TTFT 多数并发下 fusion 更优(conc64:600.93 vs 714.73;conc32:502.23 vs 551.48),但 conc4 反而略差(142.63 vs 132.84)。

## 验证与回归风险

验证为 profiler 对比(benchmark_latency.py,enable_fi_allreduce_fusion=true,+rms_norm custom op)与上述 serving 表。风险:仅影响 torch.compile 的 allreduce+norm fusion pass 的触发条件;修复后此前从未实际启用的路径开始生效,FI fused kernel 数值路径首次在用户负载中运行,精度依赖 FlashInfer kernel(参数含 fp32_acc 等不变)。

## 关联

后续:#33985(同一 fusion 迁移到 FlashInfer 统一 allreduce_fusion API)。同区域卡:#20691(sm100 FlashInfer fused allreduce 集成)、#21069(allreduce+rmsnorm+quant fusion)、#24248(torch native allreduce fusion)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
