# PR #9469: fix: tmp revert gpt oss tp sharding on hopper

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9469](https://github.com/sgl-project/sglang/pull/9469) |
| 作者 | zhyncs |
| 合并日期 | 2025-08-22 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Parallelism / Communication |
| bug 类型 | regression |

## 一句话结论
临时回退 Hopper 上的 GPT-OSS TP 切分:非 SM100 平台把 per-rank expert 中间维 block 数改回整除式计算。

## 问题现象
正文 Motivation/Modifications 等章节均为空模板,未描述具体现象。从标题可知:GPT-OSS 模型在 Hopper 平台上的 TP sharding 需要临时回退("tmp revert gpt oss tp sharding on hopper")。触发条件、报错与影响面正文均未给出。

## 根因
正文未披露根因。标题表明这是对 GPT-OSS TP sharding 在 Hopper 上问题的临时回退。(由 diff 推断)机制:改动位于 `gpt_oss.py` 的 `_load_mxfp4_experts_weights`,per-rank expert 中间维 block 数的计算改为按硬件分派——SM100(Blackwell)支持时仍用 `math.ceil(intermediate_size_block / moe_tp_size)`(向上取整切分),否则(Hopper 等非 SM100 平台)恢复为整除式 `intermediate_size_block // moe_tp_size`(向下取整)。即 Hopper 上退回旧的切分行为,ceil 切分只保留在 SM100 平台。

## 修复方案
修改 `python/sglang/srt/models/gpt_oss.py` 中 `_load_mxfp4_experts_weights` 的 per-rank 尺寸计算,按 `_is_sm100_supported` 分派:

```diff
-        per_rank_intermediate_size_block = math.ceil(
-            intermediate_size_block / moe_tp_size
-        )
+        if _is_sm100_supported:
+            per_rank_intermediate_size_block = math.ceil(
+                intermediate_size_block / moe_tp_size
+            )
+        else:
+            per_rank_intermediate_size_block = intermediate_size_block // moe_tp_size
```

后续的 `per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block` 与切片边界计算保持不变。

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间。
- 回归面:仅 GPT-OSS mxfp4 experts 权重的 TP 加载/切分路径。非 SM100(Hopper 等)平台上 per-rank expert 尺寸回到整除式计算,若 TP size 不能整除 intermediate_size_block,分配给各 rank 的 expert 容量语义会与 ceil 方案不同;SM100 平台行为不变。标题标注 "tmp",说明是临时性回退,后续可能重做 Hopper 切分方案。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
