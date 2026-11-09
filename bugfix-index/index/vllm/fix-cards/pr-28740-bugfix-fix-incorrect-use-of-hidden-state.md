# PR #28740: [Bugfix] Fix incorrect use of hidden_states for shared_experts due to do_naive_dispatch_combine

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#28740](https://github.com/vllm-project/vllm/pull/28740) |
| 作者 | alexm-redhat |
| 合并日期 | 2025-11-14 |
| labels | bug, deepseek, ready |
| 热度 | 👍 2 · 💬 1 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
DP+EP naive dispatch 后 shared_experts 误用合并后的 hidden_states;改回让 shared_experts 拿原始 hidden_states。

## 问题现象
该 PR 修复 #28406 引入的问题。触发场景(正文给出的复现配置):`VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`、`VLLM_FLASHINFER_MOE_BACKEND=latency`、`VLLM_USE_FLASHINFER_MOE_FP8=1`、`VLLM_USE_FLASHINFER_MOE_FP4=1`,以 `vllm serve nvidia/DeepSeek-R1-0528-FP4 --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel` 启动。正文未描述具体症状表现(报错或输出异常的细节),仅说明问题由 @hjjq 发现并定位到问题 commit。

## 根因
正文披露:`do_naive_dispatch_combine` 路径(DP+EP 执行)会把 `hidden_states` 经 `get_ep_group().dispatch()` 原地替换为合并后的张量,导致 shared_experts 拿到的是合并后的 hidden_states 而不是原始的——修复方向即"确保 shared_experts 拿到原始 hidden_states"。

## 修复方案
`vllm/model_executor/layers/fused_moe/layer.py` 的 `forward_impl`:dispatch 结果改存到新变量 `hidden_states_combined`,原 `hidden_states` 保持 dispatch 前的值;`quant_method.apply` 显式按路径选择输入:

```python
hidden_states_combined, router_logits = get_ep_group().dispatch(
    hidden_states, router_logits, self.is_sequence_parallel
)
final_hidden_states = self.quant_method.apply(
    layer=self,
    x=hidden_states_combined
    if do_naive_dispatch_combine
    else hidden_states,
    ...
)
```

由此 MoE 专家计算用 dispatch 后的合并输入,shared_experts 则继续使用原始 hidden_states。

## 影响与回归风险
- 影响窗口:由 #28406 引入该行为之后(正文未写明版本号)。
- 回归面:仅 `fused_moe/layer.py` `forward_impl` 的 `do_naive_dispatch_combine` 分支;不带 DP+EP naive dispatch 的路径走 `else` 分支、输入不变。影响所有经该路径的 MoE 模型(正文场景为 DeepSeek-R1-0528-FP4)。正文未附专门测试或性能数据。

## 关联
- [#28406](https://github.com/vllm-project/vllm/pull/28406)(引入问题的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
