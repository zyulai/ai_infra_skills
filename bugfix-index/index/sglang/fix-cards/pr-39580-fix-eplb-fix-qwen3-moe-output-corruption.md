# PR #39580: [Fix] eplb: Fix Qwen3 MoE output corruption after EPLB rebalance

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39580](https://github.com/sgl-project/sglang/pull/39580) |
| 作者 | sihancx |
| 合并日期 | 2026-10-20 |
| labels | (none) |
| 热度 | 👍 0 · 💬 7 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
开启 EPLB 后 Qwen3 MoE 输出尾部损坏;因 `forward_normal()` 调 TopK 缺 expert-location 派发信息,补传修复。

## 问题现象
在 4*A100 的 `Qwen3-30B-A3B`(tp-size 4 / ep-size 4)与 8*L20 的 `Qwen3-235B-A22B-Thinking-2507-FP8`(tp-size 8 / ep-size 8)上开启 EPLB(`--enable-eplb`、`--ep-num-redundant-experts 4/8`、`--eplb-rebalance-num-iterations 100`、`--expert-distribution-recorder-mode stat`、`--disable-custom-all-reduce`)时:当 prompt 的 `max_tokens` 超过 `--eplb-rebalance-num-iterations`,rebalance 发生后输出的尾部内容损坏(作者附有损坏截图)。

## 根因
正文未披露机制层面的根因分析。正文说明该问题与 #18213 所修问题同类:Qwen3 MoE 路径的 `forward_normal()` 调用 `TopK` 时没有传入 expert-location dispatch 信息,而其他 MoE 路径已在 #18213 中修复。EPLB 重平衡后需要该派发信息把逻辑 expert 映射到重排后的物理 expert 位置,缺失即产生错误输出(此机制为「由 diff 推断」)。

## 修复方案
把 #18213 引入的修复套用到 Qwen3 MoE 的同类路径:在 `python/sglang/srt/models/qwen3_moe.py` 的 `forward_normal()` 中,按 `get_exec().moe.enable_eplb` 构造 `ExpertLocationDispatchInfo` 并传给 `TopK`。关键摘录:
```python
dispatch_info = (
    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
    if get_exec().moe.enable_eplb
    else None
)
...
topk_output = self.topk(
    hidden_states,
    router_logits,
    expert_location_dispatch_info=dispatch_info,
)
```
验证:作者附上修复后的准确性测试截图(checklist 中勾选了 accuracy/speed benchmark 项)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `qwen3_moe.py` 的 `forward_normal()` 路径。EPLB 开启时 TopK 派发行为改变(增加 dispatch info),未开启 EPLB 时传 `None`,行为不变。影响 Qwen3 MoE 所有走 `forward_normal()` 的推理路径,TP/EP 配置下的正常输出依赖该信息与 EPLB 重平衡状态一致。

## 关联
- [#18213](https://github.com/sgl-project/sglang/pull/18213)(所套用的原始修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
