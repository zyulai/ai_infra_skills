# PR #32019: [LoRA][Perf] Improve FusedMoE LoRA performance for small rank

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-01-10 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32019 |

## 一句话结论

按 LoRA rank 收缩 FusedMoE LoRA Triton kernel 的 block 尺寸,避免 offs_bn 重复加载 lora 权重,gpt-oss-20b 端到端吞吐 +9%。

## 动机

FusedMoE LoRA 的 Triton kernel 默认使用与主 MoE 相同的 config,`BLOCK_SIZE_N` 默认为 64。对 shrink 操作,LoRA rank 是 N 维,`offs_bn` 用 mod 计算;当 N(rank)小于 BLOCK_SIZE_N 时 `offs_bn` 会重复,导致 lora 权重被重复加载、`tl.dot` 产生大量冗余的数据加载与计算。expand 侧同理,rank 是 K 维,过大的 `BLOCK_SIZE_K` 造成重复加载。小 rank(如 8)场景下这些冗余构成明显开销。

## 优化点

让 kernel tiling 依赖 rank:新增 `try_get_optimal_moe_lora_config` 包装原 `try_get_optimal_moe_config`,shrink 时把 `BLOCK_SIZE_N` 覆盖为 `min(原值/64, next_power_of_2(rank))`,expand 时把 `BLOCK_SIZE_K` 覆盖为 `max(16, min(原值/32, next_power_of_2(rank)))`。`FusedMoEWithLoRA._get_lora_moe_configs` 改用该 helper,并按 `op_type`(`fused_moe_lora_{w13|w2}_{shrink|expand}`)分别取 config,而不是 shrink/expand 共用一份。同时更新 `get_lora_op_configs` 中 `fused_moe_lora_*` 的默认 config:shrink `block_n=min(64, next_power_of_2(rank))`,expand `block_k=max(16, min(32, next_power_of_2(rank)))`。机制上是消除 offs 重复带来的冗余 load 与无效 dot 计算。

## 关键改动

```text
vllm/lora/layers/utils.py: 新增 try_get_optimal_moe_lora_config,按 op_type 用 next_power_of_2(rank) 覆盖 shrink BLOCK_SIZE_N / expand BLOCK_SIZE_K
vllm/lora/layers/fused_moe.py: FusedMoEWithLoRA 改用新 helper,shrink/expand 分别以 op_type 取 config
vllm/lora/ops/triton_ops/utils.py: get_lora_op_configs 中 fused_moe_lora_* 默认 config 按 rank 调整 block_n / block_k
```

关键代码摘录(节选,<=15 行):

```python
if op_type in ["fused_moe_lora_w13_shrink", "fused_moe_lora_w2_shrink"]:
    config["BLOCK_SIZE_N"] = min(
        config.get("BLOCK_SIZE_N", 64), next_power_of_2(rank)
    )
elif op_type in ["fused_moe_lora_w13_expand", "fused_moe_lora_w2_expand"]:
    config["BLOCK_SIZE_K"] = max(
        16, min(config.get("BLOCK_SIZE_K", 32), next_power_of_2(rank))
    )
```

## 性能声明

- 端到端(openai/gpt-oss-20b,TP=1,--enable-lora,max-lora-rank 8,ShareGPT 1000 prompts,output-len 300,concurrency 16,vllm bench serve):output token throughput 1533.55 → 1675.97 tok/s,+9%;Mean TPOT 10.13 → 9.22 ms;Mean ITL 10.13 → 9.22 ms;Mean TTFT 81.64 → 86.89 ms(略升)。
- kernel profile(prefill):shrink `_fused_moe_lora_kernel` 约 63µs → 42µs,expand 约 49µs → 37µs。

## 验证与回归风险

- 测试:pytest tests/lora/test_gptoss_tp.py,通过;端到端以 vllm bench serve A/B 对照。
- 影响路径:启用 LoRA 的 FusedMoE 层在无 tuned config 时的 fallback config 选择与 fused_moe_lora Triton kernel 默认值。block 尺寸变化不影响数值语义(offs 重复消除是纯效率问题),但小 rank 之外(rank >= 64 或 32)时 min/max 边界使 config 回到原值;block_shape 量化路径沿用原 config。

## 关联

- 同区域:索引库已有 pr-32655(unpermute-aware fused MoE LoRA path)、pr-23777(LoRA 启动加速)等 LoRA 性能系列;PR 正文未直接列出前序/后续。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
