# PR #3958: [Bug fixed] fixed the crash when enable the dp-attention on the single card

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3958](https://github.com/sgl-project/sglang/pull/3958) |
| 作者 | DavidChan0519 |
| 合并日期 | 2025-03-10 |
| labels | (none) |
| 热度 | 👍 2 · 💬 7 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
单卡开 dp-attention 时 crash:world_size==1 时 all_gather 返回值缺 start/end idx,改为直接返回 (tensor, 0, all_lens[0])。

## 问题现象
在单卡上启用 dp-attention 时发生 crash。正文指出:虽然单卡上开 dp-attention 没有实际意义,但该配置仍应能正常工作。正文未给出具体报错栈与更多触发细节。

## 根因
正文未单独写根因分析,但 Modifications 说明了问题所在:当 `world_size == 1` 时,deepseek_v2.py 中 DP attention 的 `all_gather` 直接 `return input_tensor`,只返回了张量本身,而调用方需要的是「输入张量、start idx、end idx」三元组(修复时 start idx 取 0)。

从 diff 可以描述改动所暗示的机制:修复前的 `world_size == 1` 早退分支位于读取 `forward_batch.global_num_tokens_cpu` 之前,返回值形状与正常路径(返回 padded tensor 及起止索引)不一致,调用方按三元组解包即崩溃。(由 diff 推断)

## 修复方案
改 `python/sglang/srt/models/deepseek_v2.py` 的 `all_gather` 函数:把 `world_size == 1` 的分支移到 `all_lens`/`max_len` 计算之后,返回 `input_tensor, 0, all_lens[0]`,与正常路径的返回签名一致。

```diff
-    if world_size == 1:
-        return input_tensor
-
     all_lens = forward_batch.global_num_tokens_cpu
     max_len = max(forward_batch.global_num_tokens_cpu)

+    if world_size == 1:
+        return input_tensor, 0, all_lens[0]
+
```

## 影响与回归风险
- 回归面:仅影响 `deepseek_v2.py` 中 DP attention 的 `all_gather` 在 `world_size == 1`(单卡)时的早退路径;多卡(world_size > 1)路径不受影响。单卡 dp-attention 属于低频配置,风险面很小。

## 关联
正文未列出
