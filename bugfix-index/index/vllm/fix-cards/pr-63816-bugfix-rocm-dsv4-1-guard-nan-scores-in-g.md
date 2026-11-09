# PR #63816: [Bugfix][ROCm][DSv4.1] Guard NaN scores in gfx942 mono routing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63816](https://github.com/vllm-project/vllm/pull/63816) |
| 作者 | frida-andersson |
| 合并日期 | 2026-10-29 |
| labels | bug, rocm, ready, deepseek, mrv2, DSv4.1 |
| 热度 | 👍 0 · 💬 22 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
gfx942 mono 路由 `fx.max` 保留 NaN 且 NaN 自比相等,同一专家被选两次致 gather 越界崩溃;NaN 排末位 + ballot 掩码越界 lane。

## 问题现象
gfx942 上,DeepSeek-V4.1-Flash 以 `--monokernel` + DSpark 自适应验证(`num_speculative_tokens=5`、`enable_adaptive_verification:true`)启动时,CUDA graph capture 期间四个 rank 全部 `Memory access fault`,engine core failed,服务无法启动。触发链条:`batches_to_profile` 是唯一向 `_dummy_run` 传非零 `context_len` 的调用者,`set_dummy_context` 用没人写过的 block id 填充 block table,draft 读到它们后 draft FFN 得到 NaN router scores。该 fault 在当时 main 上即可复现(MI325X ×4,gfx942)。

## 根因
正文披露:`stage_route` 用 `fx.max` 排名,`fx.max` 会保留 NaN;NaN 与自身比较相等,导致 draft shape 把同一 expert id 选出两次;route 用 bitmap 统计赢家,计数偏短,后续 gather 读出 slot table 之外。此外本 patch 不再包含 KV fill(正文:"The KV fill is gone")。佐证:本地排名检查显示未加防护的 NaN 行选出 `[[0, 64, 0], [0, 64, 0]]`,bitmap 28 对 30 槽;Kimi-K3 与 MiniMax-M3 已在排名前丢弃 NaN 分数,此 router(每 lane 持 2 个专家、选 3 个)还需要 ballot 掩码这一补丁特有处理。

## 修复方案
`vllm/models/deepseek_v41/amd/mono/stages/moe.py` 的 `stage_route` 两处:
1. 排名值 NaN 化为 `-inf`(`fx.isnan(val).select(neg, val)`),使 NaN 分数排最后、不再重复胜出;
2. 赢家 ballot 忽略超出专家数的 lane:`ball = ballot((cv == best) & (cursor < ept))`。
关键 diff 摘录:
```python
-            vals.append(fx.max(sc + ld_f32(a["bias"], e), fx.Float32(float("-inf"))))
+            val = fx.max(sc + ld_f32(a["bias"], e), neg)
+            vals.append(fx.isnan(val).select(neg, val))
...
-            ball = ballot(cv == best)
+            ball = ballot((cv == best) & (cursor < ept))
```
新增测试 `test_nan_draft_route_stays_in_range`(10 draft tokens / 128 experts / top-3 / 全 NaN logits):加补丁后跑通且每个 token 的 3 个 id 唯一、在界内;对 main 的 `moe.py` 同测试直接 `Memory access fault`、EXIT:134。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 gfx942 monokernel 的 `stage_route`——它在**每次** monokernel FFN launch 上运行,包括关闭自适应验证的 target-only decode。
- 回归面:仅 gfx942 mono MoE 路由的排名与 ballot 逻辑;无配置或 API 变化。有限分数选出的专家不变(本地检查:有限 draft scores 与 target 384/6 shape 加不加防护一致);`niah_single_1`@4096 前八样本与 stock 同为 1.0 且 8 个 completion 完全相同;TPOT 三档(512/128/4、1024/256/8、512/512/16)均在噪声内。验证用 nightly `0.32.1rc1.dev505+g00b6c4c56` 二进制 + main Python(未从本树重建);capture 加补丁后 60s 完成(14.30 GiB)、启动成功并可产出 completion。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
