# PR #7757: Fix incorrect spec_num_draft_tokens in draft_extend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7757](https://github.com/sgl-project/sglang/pull/7757) |
| 作者 | ch-wan |
| 合并日期 | 2025-07-05 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
draft extend 阶段 `spec_num_draft_tokens` 取值错误,在 PD 分离下触发 illegal memory access;改为 `speculative_num_steps + 1` 并补充 gather/scatter 说明。

## 问题现象
@sleepcoo 报告在 PD 分离(PD disaggregation)场景下出现 illegal memory access,本 PR 应修复该问题。正文未给出完整复现步骤或日志。

## 根因
正文未披露详细根因,仅说明该 PR 用于修复 PD 分离下的 illegal memory access。

触发机制(由 diff 推断):在 `forward_draft_extend_after_decode` 中,`spec_num_draft_tokens` 被错误地设置为 `speculative_num_draft_tokens`,而 draft extend 实际的 buffer 按每个 token `speculative_num_steps + 1` 个 draft token 位置进行 padding。新增的 diff 内注释明确写道:"During draft extend, the gathered_buffer is padded to num_tokens * (speculative_num_steps + 1). But the size of local_tokens is total accepted tokens. We need to reduce the local_num_tokens to the actual size of the accepted tokens." 即 gather buffer 的 padding 尺寸与实际 accepted tokens 数量不一致,DP gather/scatter 时若按错误的 spec_num_draft_tokens 处理就会越界访问。

## 修复方案
两处改动:

1. `python/sglang/srt/speculative/eagle_worker.py`,`forward_draft_extend_after_decode` 中修正取值:

```diff
-        model_worker_batch.spec_num_draft_tokens = self.speculative_num_draft_tokens
+        model_worker_batch.spec_num_draft_tokens = self.speculative_num_steps + 1
```

2. `python/sglang/srt/layers/dp_attention.py`:在 `_dp_gather` 与 `dp_scatter` 的 `is_draft_extend()` 分支处补上上述 NOTE 注释,说明 draft extend 下 gathered_buffer 按 `num_tokens * (speculative_num_steps + 1)` padding、而 local_tokens 为实际 accepted tokens 数,需要把 `local_num_tokens` 收缩到实际大小(该收缩逻辑本身已存在,本 PR 只是补充说明)。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围;按 diff 推断为使用 EAGLE 投机解码且开启 DP attention/PD 分离的组合场景。
- 回归面:改动核心值只作用于 draft extend 路径的 `spec_num_draft_tokens`,影响 draft model runner 的 buffer 分配与 DP gather/scatter 的 token 收缩判断;若某些后端依赖旧值 `speculative_num_draft_tokens`,行为会变化。dp_attention.py 仅新增注释,不影响运行逻辑。

## 关联
正文未列出(正文仅 @ 了报告者 sleepcoo,未附 issue/PR 链接)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
