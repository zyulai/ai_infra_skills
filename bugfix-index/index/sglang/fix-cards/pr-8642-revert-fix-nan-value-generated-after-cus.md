# PR #8642: Revert "Fix nan value generated after custom all reduce (#8532)"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8642](https://github.com/sgl-project/sglang/pull/8642) |
| 作者 | zhyncs |
| 合并日期 | 2025-08-01 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
回滚 #8532「修复 custom all reduce 后产生 NaN」的改动,把 custom_all_reduce.py 中的 torch.zeros/zeros_like 缓冲区分配恢复为 torch.empty/empty_like。

## 问题现象
本 PR 是一个 revert,正文未描述具体现象,仅说明「This reverts commit 4a6e7a66a02ac7ab966868933a59ddbb7c153c6b」,即撤销 #8532(Fix nan value generated after custom all reduce)的全部改动。被撤销的 #8532 针对的现象是 custom all reduce 之后生成 NaN。

## 根因
正文未披露回滚的原因(Motivation 与 Modifications 节均为空模板)。从 PR 标题与 diff 只能确认:本次操作是把 #8532 的改动整体撤销。(由 diff 推断)#8532 曾把 `custom_all_reduce.py` 中 `rank_data`、`buffer`、`out` 及 warm-up 返回值的分配从 `torch.empty`/`torch.empty_like` 改为 `torch.zeros`/`torch.zeros_like`(用零初始化避免读到未初始化内存而产生 NaN);本 PR 将这些位置全部改回 `torch.empty`/`torch.empty_like`,即恢复 #8532 合并前的分配方式。回滚动机正文未写,不能推断编造。

## 修复方案
改 `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`,共 6 处分配恢复为 empty 语义:`__init__` 中的 `rank_data`(两处)与 `buffer`、`all_reduce_reg`/`all_reduce_unreg`/`all_reduce` 中 `out=None` 时的 `out`、`custom_all_reduce` CUDA graph warm-up 分支的返回张量。关键 diff 摘录(同类修改之一):

```diff
-            self.rank_data = torch.zeros(
+            self.rank_data = torch.empty(
                 8 * 1024 * 1024, dtype=torch.uint8, device=self.device
             )
```

其余几处为 `torch.zeros_like(inp)` → `torch.empty_like(inp)` 与 warm-up 路径 `torch.zeros_like(input)` → `torch.empty_like(input)`。

## 影响与回归风险
- 回归面:`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` 中 custom all reduce 的缓冲区/输出张量分配路径(注册、非注册、CUDA graph 内外)。改动后这些内存回到「未零初始化」状态;若 #8532 所针对的 NaN 场景仍然成立,该风险会随之恢复(正文未验证说明)。使用 custom all reduce 的多卡推理路径均受影响。

## 关联
- https://github.com/sgl-project/sglang/pull/8532 (被 revert 的 PR,见标题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
