# PR #28449: [Feat][Perf] Enable deepep-low-latency with round-robin expert placement.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cboss6 |
| 合并 | 2025-11-19 |
| 分类 | MoE(major) |
| Label | ready |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/28449 |

## 一句话结论

让 round-robin 专家放置策略可在 DeepEP low-latency all2all 后端下工作,DeepSeek-R1 双节点 H20 吞吐 +14.57%、TPOT 改善 13.38%。

## 动机

此前 round-robin 专家放置与 DeepEP low-latency all2all 后端无法组合:DeepEP-LL 的 dispatch/combine 直接以 topk_ids 作为 expert id,隐含 linear placement 的“全局 id 与物理 id 线性对应”假设;round-robin 交错放置时两者不一致,导致低延迟路径只能用默认 linear 放置,负载与通信不均。

## 优化点

在 FusedMoE 层引入三张 routing 表并由 DeepEP-LL prepare/finalize 在运行时做 id 重映射:`ensure_round_robin_expert_routing_tables` 按 `owner = gid % ep_size` 计算 `global_to_physical`(含余数rank的偏移修正)、逆表 `physical_to_global` 和本地表 `local_expert_global_ids`;dispatch 与 combine 前用 `global_to_physical[topk_ids]` 把全局专家 id 翻译成物理 id。同时把放置策略判定抽成 `determine_expert_placement_strategy`:round_robin 仅在多 expert group、无冗余专家、非 EPLB 且后端为 DeepEP-LL 时生效,否则 warning 后回退 linear,避免错误组合。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: determine_expert_placement_strategy、_maybe_init_expert_routing_tables、ensure_round_robin_expert_routing_tables
vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py: _map_global_to_physical_ids,dispatch/combine 用映射后的 id
vllm/model_executor/layers/fused_moe/all2all_utils.py: maybe_make_prepare_finalize 新增 routing_tables 参数并传入 DeepEP-LL
各 quant method(fp8/modelopt/compressed_tensors): maybe_make_prepare_finalize 签名透传 routing_tables
```

dispatch 处的核心映射:

```python
dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
expert_x, expert_num_tokens, handle, _, hook = self.buffer.low_latency_dispatch(
    a1, dispatch_topk_ids, ...)
```

## 性能声明

测试平台 CUDA 12.8、driver 550.144.03;模型 DeepSeek-R1-671B,H20×8×2 nodes;配置 dp=16、tp=1、enable_expert_parallel、all2all_backend=deepep_low_latency、use_deep_gemm=1。benchmark 条件:sharegpt 数据集,num_prompts=512,input_len=1024,output_len=512,max_concurrency=8,req_rate=8,times=10。对比默认 linear placement:吞吐提升 14.57%,TPOT 改善 13.38%。精度:GSM8K 5-shot exact_match 0.97±0.0171(flexible-extract 与 strict-match 相同)。

## 验证与回归风险

PR 正文只给出 GSM8K 精度评估与 serving benchmark,未列出单元测试新增。风险面:`maybe_make_prepare_finalize` 签名变更波及多个 quant method 实现路径;routing 表为注册 buffer,`update_expert_map` 后会重建;round_robin 组合不满足约束时自动回退 linear,未启用 DeepEP-LL 的路径行为不变(topk_ids 映射在表为 None 时直通)。

## 关联

依赖既有的 round_robin 专家放置与 DeepEP low-latency backend 机制,本 PR 打通两者组合;同分类其他 PR 及前序/后续编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
