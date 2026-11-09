# PR #34189: [DSV4] Fix silent KV corruption when speculative draft tokens > 4

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34189](https://github.com/sgl-project/sglang/pull/34189) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-08-09 |
| labels | high priority, deepseek, run-ci, jit-kernel, bypass-fastfail |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
DSV4 压缩写计划把 pad 硬编码为 4,draft tokens > 4 时环形区欠写、后续压缩读到陈旧槽位,静默产生错误压缩状态;改为按环形容量推导 pad 并在启动时校验。

## 问题现象
DeepSeek-V4 投机解码在 `--speculative-num-draft-tokens > 4` 时,压缩状态被静默破坏——正文强调「no illegal access, no NaN, no assertion, just wrong compressed state」,完全无显性症状。端到端数字(GSM8K 0.990→0.985、GPQA 0.7172→0.7374)看不出回归,作者解释:被跳过的位置是 draft tail 头部,相邻 verify batch 写集高度重叠,下一步通常会重写上一步跳过的位置,腐坏需要「位置一直未写直到压缩恰好读它」,依赖 accept 长度、序列长度与压缩点对齐。kernel 级回归测试对修复前 planner 跑出 19 个失败(c4 在 D=5 即失败,边界正好是 D > 4)。

## 根因
正文披露:压缩写计划决定哪些 token 的 raw KV 写入 compress-state 环形区供未来压缩窗口读取;投机 verify batch 按乐观 `seq_len = prefix_len + num_draft_tokens` 规划,再回滚到 `prefix_len + accept_len`,因此所有已提交 token 无论 accept 长度如何都必须驻留。`c_plan.cuh` 把 pad 硬编码为 `kMaxMTPDraftTokens = 4`(`std::min(ring_size - compress_ratio, 4)`),draft tokens > 4 时计划欠写环形区。此外 host planner 路径完全缺 pad 项,CPU 与 GPU 两个 planner 输出不同写集(独立缺陷)。

## 修复方案
1. `python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` `plan_compress_prefill`:删除 `kMaxMTPDraftTokens = 4` 常量,pad 改为环形区属性 `mtp_pad = ring_size > window_size ? ring_size - window_size + 2 : 0`(推导:写入 `w` 别名到 `w - ring_size`,未来压缩最远需要 `prefix_len - window_size + 2`);三个 planner 路径(kernel0 两个分支与 C++ host loop)共享同一值,host loop 的 `first_w_pos` 加 `std::min(..., seq_len - mtp_pad)` 修正;
2. `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`:新增 `get_compress_state_write_pad()` 镜像该上界;
3. `python/sglang/srt/model_executor/pool_configurator.py` `DSV4PoolConfigurator`:启动时 `_assert_ring_serves_draft_tokens` 按 `max_speculative_num_draft_tokens` 拒绝环形区无法服务的 draft 数——planner 无法区分超配 verify batch 与普通长 prefill,故在启动时失败而非运行时腐坏。当前环形区容量:c4 支持 num_draft_tokens ≤ 10,c128 ≤ 130;扩大 c4 上界需增大 c4 状态池,本 PR 未做。非投机环形区恰为一个 window 宽,pad 为 0,写集不变。

```diff
-  constexpr int32_t kMaxMTPDraftTokens = 4;
-  const auto mtp_pad = std::min(ring_size - compress_ratio, kMaxMTPDraftTokens);
+  const auto mtp_pad = ring_size > window_size ? ring_size - window_size + 2 : 0;
```

## 影响与回归风险
- 影响窗口:正文未说明引入版本;触发条件为 DSV4 投机解码且 `speculative-num-draft-tokens > 4`(c4 层暴露最集中;c128 仅在接近容量时受影响)。
- 回归面:DSV4 compress write plan 的三条 planner 路径(GPU kernel0 两分支 + CPU host loop)、投机 verify batch 的写入行数(比修复前更多,受环形容量约束)、启动时 pool 配置校验(超配服务器由静默腐坏改为启动失败);非投机服务的写集经测试确认不变。无性能影响:常数换成 planner 已有值的表达式,无额外内存流量与新 kernel 启动。

## 关联
- https://github.com/sgl-project/sglang/pull/33872 (被本 PR 取代)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
