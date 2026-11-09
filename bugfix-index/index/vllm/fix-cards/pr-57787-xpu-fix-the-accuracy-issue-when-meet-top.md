# PR #57787: [XPU]Fix the accuracy issue when meet topk_ids=-1 on DP+EP scenarios

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57787](https://github.com/vllm-project/vllm/pull/57787) |
| 作者 | yisustc |
| 合并日期 | 2026-10-09 |
| labels | intel-gpu |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

DP+EP 下 dummy/padded token 的 topk_ids=-1 被 XPU MoE kernel 直接当索引用,越界读污染真实 token 输出并经 combine 扩散;EP 时把负 id 重映射到本 rank 不拥有的专家。

## 问题现象

DP + EP 运行 MoE 模型时,dummy/padded token(例如空闲 DP rank 执行的 dummy batch)被标记为 `topk_ids = -1`;EP dispatch 把这些 -1 条目 all-gather 到每个 rank 的 fused MoE kernel。UT `v1/distributed/test_external_lb_dp.py::test_external_lb_completion_streaming[api-1-ep-enabled-ibm-research/PowerMoE-3b]` 失败:流式与非流式输出不一致,断言 `' (-RelatedRelated BR' == ' Share ( JavaScript CloseAn'` 失败。

## 根因

正文披露:XPU MoE kernel 直接把 `topk_ids` 当索引用,-1 造成越界读(未定义行为),不仅破坏 dummy 行,还破坏真实 token 的输出,产生非确定结果与偶发 NaN,并经 EP combine 扩散到所有 rank。正文同时指出 DeepEP HT 等其他平台路径(指向 `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py#L225`)已有同样处理,避免让 topk_ids=-1 参与计算。

## 修复方案

改 `vllm/model_executor/layers/fused_moe/experts/xpu_moe.py` 的 `apply`:当 `ep_size > 1` 时,把负的 topk_ids 重映射为哨兵值,使其指向本 rank 不拥有的专家从而被跳过:

```python
if self.moe_config.ep_size > 1:
    # Avoid topk_ids = -1, remap them to an
    # expert not owned by this rank so they are skipped.
    sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
    topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
```

即 ep_rank 0 用 `global_num_experts - 1`,其余 rank 用 0 作为哨兵。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;触发条件为 XPU + MoE + DP+EP(EP 维度存在 dummy/padded token)。
- 回归面:仅 XPU MoE 的 EP(`ep_size > 1`)路径,非 EP 场景与非 XPU 平台不变。哨兵选择隐含假设 ep_rank 0 不拥有最后一个专家、其他 rank 不拥有专家 0(diff 注释称其为「本 rank 不拥有的专家」),若专家划分方式变化使该假设不成立,dummy token 会被实际计算而非跳过(影响性能而非正确性);dummy 行的输出语义由未定义变为确定计算后丢弃,真实 token 路径的 topk_ids 均非负,不受影响。

## 关联

正文未列出(正文以链接引用了 DeepEP HT 的 deepep_ht.py 处理位置)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
