# PR #31311: Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31311](https://github.com/sgl-project/sglang/pull/31311) |
| 作者 | whn09 |
| 合并日期 | 2026-07-20 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 1 |
| 分类 | MoE |
| bug 类型 | crash |

## 一句话结论
LongCat-2.0 在 deepep real EP 下有两个耦合 bug:MoE 前向多打一次 TP all-reduce 导致输出被放大约 tp_size 倍成乱码,ScMoE 稠密分支拿到散射输入后在 RoPE reshape 崩溃;分别用门控跳过与 all-gather 对齐修复。

## 问题现象
LongCat-2.0 在真实 MoE all-to-all 后端(`--moe-a2a-backend deepep`,tp16/ep16)下崩溃或输出乱码;两个 bug 都只在 real-EP 路径出现(`a2a=none` 时为 no-op):
1. 输出"连贯但是乱码"(coherent-but-garbage):MoE 分支输出被放大约 `tp_size` 倍;
2. 直接崩溃:RoPE view 处报 `shape '[16,-1,64]' invalid for input of size 256`,随后在最终 norm 处还有 branch-merge 形状不匹配。
正文 Accuracy 验证环境:`meituan-longcat/LongCat-2.0-FP8`,tp16/ep16 over EFA;只修 RoPE 不修 all-reduce 时仍输出乱码,两者都修后 EN/ZH/math prompt(deepep `normal` 与 `low_latency` 两种模式)输出连贯,norm 探针显示 MoE 分支幅度从约 4-5(虚高)回落到约 1.1(与 `a2a=none` 正确基线一致)。

## 根因
正文披露两个根因:
1. **`LongcatFlashMoE.forward` 双重 all-reduce**:`forward()` 以无条件的 post-experts `tensor_model_parallel_all_reduce` 结尾。这对 `a2a=none`(EP-over-TP:experts 是 TP 分片的部分和)正确;但在真实 a2a 后端下 `self.experts` 是 `DeepEPMoE`,其 forward 已完成 `dispatch -> run_moe_core -> combine`,combine 已在 EP rank 间归约了专家输出。额外的 TP all-reduce 把已合并结果再求和一次,MoE 分支输出膨胀约 `tp_size` 倍。DeepSeek-V2 对同样的 all-reduce 用 `should_skip_post_experts_all_reduce()` 做了门控,LongCat 的单一 MoE 路径从未加门控。
2. **ScMoE 稠密分支 RoPE 崩溃 / 分支合并错位**:ScMoE 稠密分支(`forward_mlp`:`mlps[0] -> self_attn[1] -> mlps[1]`)是纯 TP,必须在复制后的完整 token 上运行(每个 mlp 自带 all-reduce;`self_attn[1]` 的 RoPE 需要完整 `positions`)。真实 a2a 后端下 moe 分支把 hidden 散射到每 rank 1 个 token,`forward_mlp` 收到散射输入,RoPE view 处形状不合法即崩溃。

## 修复方案
改 `python/sglang/srt/models/longcat_flash.py`:
1. `LongcatFlashMoE.forward`:post-experts all-reduce 增加 `get_moe_a2a_backend().is_none()` 门控,仅在 EP-over-TP 路径执行:
```diff
-        if self.tp_size > 1:
+        if self.tp_size > 1 and _lc_gab().is_none():
             final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```
2. 新增辅助函数 `_scmoe_align_rows`:把 `[rows,H]` 张量按 attn-tp 组对齐到目标行数(目标更大时 all_gather,更小时取本 rank 连续段);
3. `LongcatFlashDecoderLayer.forward_mlp`:真实 a2a 后端激活时(`not is_none() and attn_tp_size > 1` 且行数匹配 `positions`),在 `mlps[0]` 前把稠密分支 `hidden`(+`residual`)all-gather 到完整 token;分支合并处再把 `hidden_states`/`residual` 与 moe 分支 `moe_hidden_states` 的行数对齐后相加。
两条路径均有门控,`a2a=none` 行为不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `longcat_flash.py`。影响 LongCat-Flash 系模型的 MoE 前向与 ScMoE 层:real-EP(deepep 等)路径的数值结果改变(修复虚高),稠密分支新增 all-gather/切片通信开销;`a2a=none` 路径被显式门控保持原行为。`_scmoe_align_rows` 的对齐在 `attn_tp_size > 1` 且行数匹配时才触发,其他并行配置不受影响(由 diff 推断)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
