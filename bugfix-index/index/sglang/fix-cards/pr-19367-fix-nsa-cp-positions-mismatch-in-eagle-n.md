# PR #19367: Fix NSA CP positions mismatch in eagle NextN model

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19367](https://github.com/sgl-project/sglang/pull/19367) |
| 作者 | alisonshao |
| 合并日期 | 2026-02-26 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
CP 开启时 eagle NextN 模型切分 hidden_states 却不切分 positions,NSA indexer 旋转位置编码形状不匹配崩溃;补上 `cp_split_and_rebuild_position` 修复。

## 问题现象
开启 context parallelism(CP)后,eagle 投机解码(NSA + CP,ATTN_CP4 TP4 配置)运行时报错:
```
RuntimeError: shape '[8, -1, 64]' is invalid for input of size 64
```
出错位置为 `rotary_embedding/base.py:292`(完整 trace 见正文附的 GitHub Actions 链接)。

## 根因
正文披露:启用 CP 时,eagle NextN 模型(`deepseek_nextn.py`)把 `hidden_states` 按 CP rank 切分,但**没有**同步切分 `positions`,造成 NSA indexer 旋转位置编码处的形状不匹配。机制是:旋转位置编码用 `positions.size(0)` 作为 batch size 去 reshape `k_rope`,而 `k_rope` 的元素数更少——因为 `hidden_states` 已经被 CP 切分过。

## 修复方案
`python/sglang/srt/models/deepseek_nextn.py` 的 `forward` 中,在 `nsa_use_prefill_cp` 分支内给 `hidden_states` 的 `cp_split_and_rebuild_data` 调用补上配套的 positions 切分,与主模型(`deepseek_v2.py:2707`)已有的做法一致:
```diff
         if nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
             hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
+            positions = cp_split_and_rebuild_position(forward_batch, positions)
```
正文 Test plan 列出「以 NSA + CP 开启(ATTN_CP4 TP4)运行 eagle 投机解码」为待执行项(未勾选)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `deepseek_nextn.py` 的 NSA prefill-CP 路径;`positions` 在该分支被切分后与 `hidden_states` 保持一致,进入 decoder 与 NSA indexer。非 CP / 非 NSA 配置不走该分支,不受影响;NextN 之外的模型无改动。

## 关联
- 正文附完整报错 trace:[GitHub Actions run 22376709996](https://github.com/sgl-project/sglang/actions/runs/22376709996/job/64768484941)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
