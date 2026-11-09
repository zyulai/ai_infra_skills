# PR #3592: fix high qps crash when enable mtp

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3592](https://github.com/sgl-project/sglang/pull/3592) |
| 作者 | zhyncs |
| 合并日期 | 2025-02-15 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
高 QPS 开启 MTP(NEXTN)时崩溃;forward batch 初始化对 NEXTN 跳过 triton 位置计算路径。

## 问题现象
正文未描述具体现象(Motivation 与 Modifications 均留空),仅标题指出「启用 MTP 且高 QPS 时崩溃」。结合修复位置看,受影响的是启用 NEXTN/MTP 投机解码、attention 后端非 torch_native 时的 extend 预填充路径(由 diff 推断)。

## 根因
正文未披露根因(动机与修改说明均为空)。(由 diff 推断)`init_new` 原逻辑只要 attention 后端不是 torch_native,就会为 extend batch 设置 `extend_num_tokens` 并调用 `compute_position_triton` 计算 positions;启用 NEXTN(MTP)投机解码时该位置计算与投机解码的 token 布局不匹配,推测高并发(高 QPS)下由此触发崩溃,修复即对该算法跳过该分支。

## 修复方案
改 `python/sglang/srt/model_executor/forward_batch_info.py` 的 `init_new`:条件中追加 `and model_runner.server_args.speculative_algorithm != "NEXTN"`,使 NEXTN(MTP)场景不再设置 `extend_num_tokens`、不再调用 `compute_position_triton` 计算 positions。

```diff
             if (
                 model_runner.server_args.attention_backend != "torch_native"
                 and model_runner.server_args.speculative_algorithm != "NEXTN"
             ):
                 ret.extend_num_tokens = batch.extend_num_tokens
                 positions, ret.extend_start_loc = compute_position_triton(
                     ret.extend_prefix_lens, ret.extend_seq_lens, ret.extend_num_tokens
```

## 影响与回归风险
- 回归面:改动只影响 `init_new` 中这一个条件分支:当投机算法为 NEXTN 且后端非 torch_native 时,不再执行该分支内的位置计算;其他投机算法与 torch_native 后端路径行为不变。NEXTN 路径所需的 positions/extend_num_tokens 改由其余逻辑给出(由 diff 推断),若该假设不成立会影响 MTP 正确性,这正是本次修复针对的核心路径。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
