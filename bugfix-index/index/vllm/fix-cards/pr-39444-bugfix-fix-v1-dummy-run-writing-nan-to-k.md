# PR #39444: [Bugfix] Fix V1 dummy run writing NaN to KV cache null block

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39444](https://github.com/vllm-project/vllm/pull/39444) |
| 作者 | elvircrn |
| 合并日期 | 2026-04-10 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
V1 `_dummy_run` 的 slot_mapping 全 0 导致 idle DP rank 把 NaN 写进 null block(块 0)并污染精度,改为填 -1 跳过 KV 写入。

## 问题现象
影响所有使用 DP+EP 的 V1 部署(如 DeepSeek R1/V3 且 `data_parallel_size > 1`、`expert_parallel_size > 1`),与量化方式无关,FP8、NVFP4、BF16 全部受影响。在 DeepSeek R1 NVFP4 上实测精度从 97% 掉到 55%。机制上,idle DP rank 的 `_dummy_run` 走 KV 写路径,把垃圾 hidden states(常为 NaN)写入块 0;块 0 被所有未使用的 block table 条目引用,下游 attention 在 padding 位置从块 0 读取,NaN 随模型传播。测试通过在真实 DeepSeek R1 NVFP4 部署上加 KV cache NaN 埋点验证:修复前 `slot_mapping=[0]`,修复后 `slot_mapping=[-1]`,块 0 的 NaN 消失。

## 根因
正文披露:由 #25954(2026-01-24)引入——该 PR 把 slot mapping 重构为独立的 `_get_slot_mappings()`,并在 `_dummy_run` 中无条件调用、经 `set_forward_context(slot_mapping=...)` 传入。`_get_slot_mappings` 读的是用 `torch.zeros` 初始化的持久 `slot_mapping` 缓冲,真实 token 区间 `[0:num_tokens_unpadded]` 保持 0,即指向 block 0 offset 0(null block);`concat_and_cache_mla` 内核看到 `slot_idx=0 >= 0` 便执行写入。在该 PR 之前,slot mapping 只在 `_build_attention_metadata` 内(以 FULL cudagraph 模式为条件)设置,dummy run 不总会触发 KV 写。

## 修复方案
`vllm/v1/worker/gpu_model_runner.py` 的 `_dummy_run`:调用 `_get_slot_mappings` 返回后,把所有 group 的 slot mapping 填充为 -1(PAD_SLOT_ID),使 `concat_and_cache` 内核整体跳过 KV 写;同时把该调用的 `num_tokens_padded` 实参由 `num_tokens` 修正为 `num_tokens_padded`。关键摘录:

```python
# Dummy runs have no real slot assignments — fill with -1 so
# concat_and_cache kernels skip the KV write.
if slot_mappings_by_group is not None:
    for sm in slot_mappings_by_group.values():
        sm.fill_(-1)
```

与 V2 已有的 `get_dummy_slot_mappings`(其中 `self.slot_mappings.fill_(PAD_SLOT_ID)`)行为对齐。

## 影响与回归风险
- 影响窗口:#25954 合入(2026-01-24)之后、本 PR 之前的 V1 DP+EP 部署。
- 回归面:仅 `vllm/v1/worker/gpu_model_runner.py` 的 `_dummy_run` 路径(空闲 DP rank、CUDA graph 捕获等 dummy 场景);dummy run 不再产生任何 KV 写入,真实 run 的 slot mapping 逻辑未动。

## 关联
#25954(根因引入 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
