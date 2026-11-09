# PR #30333: [AMD] Fix DeepSeek V4 MTP accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30333](https://github.com/sgl-project/sglang/pull/30333) |
| 作者 | At1a8 |
| 合并日期 | 2026-07-07 |
| labels | amd, deepseek, run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
AMD 上 DeepSeek V4 MTP 首跑 GSM8K 精度掉点:C128 状态池冷启动含 torch.empty 垃圾,改为整池初始化为空态哨兵。

## 问题现象
AMD/ROCm 平台上 DeepSeek V4 开启 MTP 投机解码时,首次运行 GSM8K 评测出现精度下降(该现象在 #30238 的评论中被报告)。触发条件:服务冷启动(cold server startup)后第一次跑请求,此时 request-scoped 的 C128 状态行尚未被任何请求写过,仍是 `torch.empty()` 的未初始化内存;MTP target-verify 基于已提交的 request 状态写 draft-token 的 C128 状态,读到垃圾数据即腐蚀 draft 验证。修复后 GSM8K(1319 题)Accuracy 0.952,Invalid 0.000。

## 根因
正文明确披露:PR #28612 之后,C128 state 从 SWA-slot-scoped 改为 request-scoped。冷启动时状态池只有 sentinel 行(最后一行)被初始化,普通 request-scoped 的 C128 行仍可能包含 `torch.empty()` 分配出的垃圾数据;MTP target-verify 会依据已提交的 request 状态写 draft-token 的 C128 状态,因此未初始化的已提交行会腐蚀 draft 验证,造成首跑 GSM8K 精度下降。

## 修复方案
改 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` 的池初始化逻辑:对非 online 模式,当 `_is_hip` 且 ratio==128 时,把整个非 online C128 `kv_score_buffer` 初始化为空态哨兵(`clear()` 整个 buffer);否则保持原有只清最后一行的行为。C4 保持历史的 last-row-only 哨兵初始化,避免不必要的行为变化;`_is_hip` 守卫保证改动只作用于 AMD/ROCm,其他平台为 no-op。

```diff
         if not online:
-            self.kv_score_buffer[-1].clear()
+            if _is_hip and ratio == 128:
+                self.kv_score_buffer.clear()
+            else:
+                self.kv_score_buffer[-1].clear()
```

## 影响与回归风险
- 影响窗口:PR #28612 将 C128 state 改为 request-scoped 之后引入,直到本 PR 修复。
- 回归面:仅影响 AMD/ROCm(`_is_hip`)上非 online 的 ratio==128 C128 状态池初始化;整池初始化会在建池时多一次写操作(正文实测性能几乎无影响:TP8 DP8、8k/1k、并发 256 下吞吐 29615.05 → 29587.03 tok/s,TTFT/TPOT 基本持平);C4 与非 HIP 平台行为不变。

## 关联
- https://github.com/sgl-project/sglang/pull/28612 (C128 state 改为 request-scoped 的 PR)
- https://github.com/sgl-project/sglang/pull/30238#issuecomment-4891010577 (GSM8K 精度下降现象报告)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
