# PR #28369: [EPLB] Refactor balance_packing to use numpy and optimize GPU-CPU transfers in EPLB

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @SageMoore |
| 合并 | 2025-11-11 |
| 分类 | Model-specific Fast Path |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/28369 |

## 一句话结论

EPLB 重平衡 CPU 侧改 numpy、GPU-CPU 拷贝移出逐层循环,rebalance GPU 时间 420ms → 260ms,P99 ITL -45%。

## 动机

EPLB(专家并行负载均衡)的重平衡在每次 step 触发时,需要把专家负载统计从 GPU 拷到 CPU 做装箱计算。`balance_packing` 用 torch tensor 在 CPU 上逐层逐 group 做 Python 循环,tensor 开销叠加;`rearrange_expert_weights_inplace` 则在每层 `shuffle_layer` 循环里都 `.tolist()` 一次全局专家索引,每层都发生一次 GPU-CPU 同步拷贝。这些 CPU 开销发生在推理的 CUDA stream 上,形成 GPU bubble——在 DeepSeek-V2-Lite DP8 下 `rebalance_experts` 开销约 80ms。

## 优化点

两处。其一,`balanced_packing` 的装箱主体改为 numpy:负载 `weight.cpu().numpy()` 后用 `np.argsort(-weight)` 取降序索引,逐层贪心装箱(选当前最轻且有容量的 pack)全部操作 numpy int64 数组,最后 `torch.from_numpy(...).to(device)` 回传;`rebalance_experts` 去掉 `.cpu()`(只保留 `.float()`,由内部子函数决定拷贝时机)。其二,`rearrange_expert_weights_inplace` 把 `old/new_global_expert_indices` 的 `.cpu()` 拷贝提到 `shuffle_layer` 循环之前一次性完成,每层只做 CPU list 转换,不再逐层触发设备拷贝;原本每层的 `torch.cuda.synchronize()` 移到循环前一次(作者保留"不清楚为何需要同步"的原注释)。

## 关键改动

```text
vllm/distributed/eplb/rebalance_algo.py: balanced_packing 改 numpy 装箱,rebalance_experts 去除强制 .cpu()
vllm/distributed/eplb/rebalance_execute.py: 全局专家索引一次性拷 CPU,逐层 tolist 从 CPU 张量读取
```

关键片段(装箱改 numpy):

```python
weight_np = weight.cpu().numpy()
indices_np = np.argsort(-weight_np, axis=-1)
pack_index_np = np.full((num_layers, num_groups), -1, dtype=np.int64)
rank_in_pack_np = np.full((num_layers, num_groups), -1, dtype=np.int64)
for i in range(num_layers):
    pack_weights = [0.0] * num_packs
    pack_items = [0] * num_packs
    for group in indices_np[i]:
        pack = min((j for j in range(num_packs)
                    if pack_items[j] < groups_per_pack),
                   key=pack_weights.__getitem__)
        ...
```

## 性能声明

- 重平衡开销:8xH100、deepseek-ai/DeepSeek-V2-Lite、DP8 下 `rebalance_experts` 开销约 80ms → 约 10ms;两处改动合计使 EPLB rebalance 的 GPU 时间从 420ms 降至 260ms;`rebalance_expert_weights_inplace` 每层省约 1.5ms GPU 时间。
- 端到端:DeepSeek-V2-Lite,8 GPU,`VLLM_ALL2ALL_BACKEND=deepep_low_latency`,DP8 + expert parallel + EPLB(window 100/step 100),sharegpt 1000 prompts:Output token throughput 4968.59 → 5143.07 tok/s;P99 ITL 551.66 → 292.02 ms(约 -45%,作者称该指标代表 EPLB 开销的最坏情况);Median TPOT 69.74 → 65.78 ms;P99 TPOT 1558.07 → 1555.28 ms;TTFT 基本持平(7725.04 → 7726.39 ms)。

## 验证与回归风险

测试:正文用 CUDA stream 截图(改动前后总时长对比)与 e2e serving benchmark 验证;pack 的 diff 中未含新增测试文件。精度:DP2 + EPLB 下 lm_eval GSM8K 5-shot:flexible-extract 0.3633 / strict-match 0.3600(正文仅给出该结果,未附 before 数值)。回归风险:仅影响启用 EPLB 的部署;装箱算法语义不变(numpy 与 torch 排序的 tie-breaking 顺序可能不同,作者未专门讨论);`np.argsort(-weight_np)` 要求 weight 可转 numpy(float32);CPU 侧同步点位置变化对其他 executor 路径的影响未单独报告。

## 关联

同区域:EPLB 组件本身(rebalance_algo/rebalance_execute,DeepEP low-latency all2all 后端);pack 未列出前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
