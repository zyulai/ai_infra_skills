# PR #33614: [Spec] Fix Dspark and Dflash state divergence across TP rank

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33614](https://github.com/sgl-project/sglang/pull/33614) |
| 作者 | JackZeng0208 |
| 合并日期 | 2026-08-30 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 29 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
TP>1 时 Dspark/Dflash 各 rank 采样决策无同步会漂移并死锁,新增 SpecTpSync 在决策影响状态前广播 rank0 结果。

## 问题现象
与 issue #33289 对应。TP>1 时 SGLang 默认跳过采样 token 的跨 rank 同步(为保住投机解码加速),Dspark 在每个 rank 上独立做三类采样决策:draft Markov chain 逐步采样 proposal token(含 in-graph philox noise)、target verify 推导 `correct_len`/`bonus`/`cap_trim_lens`、prefill 采样 `next_token_ids`,没有任何机制强制各 rank 一致。作者用 fault injection 演示:让 rank 1 的 accept length 系统性少 1,未修复时 server 在 startup warmup 完成前就死锁;修复后同样注入无害,server 正常启动且 temp=0 请求返回正确输出。

## 根因
正文披露:一旦某个 rank 提交了不同的 token 或 accept length,序列长度与 KV state 就会各 rank 漂移,后续 collective 死锁。作者还解释了为什么不改成对齐 RNG seeds:各 rank 可能消耗不同数量的随机数,RNG 对齐仍会漂移,且覆盖不了非 RNG 差异。

## 修复方案
新增 `python/sglang/srt/speculative/spec_tp_sync.py` 的 `SpecTpSync`,用 TP group 的 PyNCCL communicator 在决策生效前广播 rank0 结果(DP attention 下用 `attn_tp_group`,与原 `verify_lens_broadcast_group` 一致;TP=1 是 no-op;进程组 collective 无法被 CUDA graph 捕获,PyNCCL 可以)。具体:1) eager 与 graph-folded 两条采样路径都逐 token 广播 Markov proposal;2) eager verify 与 captured verify epilogue 中广播 `correct_len`/`bonus`/`cap_trim_lens`;3) 广播 prefill `next_token_ids`。新增环境变量 `SGLANG_SPEC_TP_SYNC`(默认 all)可按站点开关。dspark_planner.py 删除了旧的 `verify_lens_broadcast_group`,统一走新机制。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:每个 decode step 增加 `gamma + 3` 次 `[bs]` tensor 广播(都在 captured graphs 内),有吞吐开销;广播依赖 PyNCCL communicator 的集合顺序,正文指出该 workload 需要 NCCL 2.30.7(torch 自带 2.28.9 会在 graph/eager 混用下卡死,属另一独立问题);cuda graph capture 的可用显存判断改为 group 内取最小值,各 rank 折叠决策一致;dflash 验收路径被重构进 `_accept_block` 并插入同步点。

## 关联
- https://github.com/sgl-project/sglang/issues/33289
- https://github.com/sgl-project/sglang/pull/33298
- https://github.com/sgl-project/sglang/issues/29003
- https://github.com/sgl-project/sglang/pull/31478
- https://github.com/sgl-project/sglang/pull/33407

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
