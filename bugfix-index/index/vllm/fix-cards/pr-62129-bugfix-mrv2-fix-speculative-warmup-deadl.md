# PR #62129: [Bugfix][MRV2] Fix speculative warmup deadlock with pipeline parallelism

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62129](https://github.com/vllm-project/vllm/pull/62129) |
| 作者 | chenjr97 |
| 合并日期 | 2026-10-18 |
| labels | bug, mrv2, ready |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
MRV2 投机解码的贪心 warmup 只在末级 PP stage 执行,其余 stage 不参与致 PP 启动死锁;改按 speculative_config 门控。

## 问题现象
MRV2(MTP)投机解码 + pipeline parallelism 的部署在启动 warmup 阶段死锁:#59260 引入的额外贪心 warmup pass 只有最后一个 PP stage 执行,前面的 stage 跳过该 pass,而末级 stage 在等待它们的激活,send/receive 无法配对,启动被挂起。触发条件:PP>1 且启用 MRV2 投机解码。

## 根因
正文披露:只有最后一个 PP stage 持有 rejection sampler,而此前的代码把"额外贪心 warmup pass"门控在 rejection sampler 是否存在上;于是非末级 stage 跳过该 pass,末级 stage 却在等它们的激活,造成死锁。

## 修复方案
修改 `vllm/v1/worker/gpu/warmup.py` 的 `warmup_kernels`:把额外贪心 warmup pass 的条件从 `if rejection_sampler is not None` 改为 `if model_runner.speculative_config is not None`,确保每个 PP stage 执行数量一致的前向(所有 rank 的 send/receive 匹配);并新增回归测试 `test_kernel_warmup_runs_same_rounds_on_every_pp_stage`,覆盖投机开关两种情况下各 PP stage 的 warmup 轮数一致。

```diff
         _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
-        if rejection_sampler is not None:
+        # All PP ranks must run the same forwards to match sends and receives.
+        if model_runner.speculative_config is not None:
```

## 影响与回归风险
- 影响窗口:自 #59260 引入该 warmup 门控起,至本修复合并止。
- 回归面:仅 warmup 条件与测试;非 PP 或未启用投机解码的路径(`speculative_config is None`)行为不变。启用投机的单机/TP 部署从"有 sampler 才多跑一轮"变为"有投机配置就多跑一轮",对持有 sampler 的 rank 等价;PP 各 stage 现在都执行贪心 warmup,前向次数增加一轮但换来 send/receive 配对。验证:34 个测试通过,GLM-4.7-Flash MRV2 TP=2 PP=2 冒烟通过(生成 1024 token,532 draft/492 accepted);未做精度基准。

## 关联
- https://github.com/vllm-project/vllm/pull/59260

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
