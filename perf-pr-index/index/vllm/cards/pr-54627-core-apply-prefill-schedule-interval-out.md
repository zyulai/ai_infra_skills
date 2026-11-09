# PR #54627: [Core] Apply prefill_schedule_interval outside data parallelism

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fululi12 |
| 合并 | 2026-10-15 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, scheduler |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/54627 |

## 一句话结论

让 prefill_schedule_interval 在单引擎 DCP 也生效，Kimi-K3 8x MI355X p90 交互性 +44.7%/+44.3%。

## 动机

`prefill_schedule_interval` 把新 prefill 推迟到每第 N 个 engine step，使 decode 以更长不间断区间运行，而非大多数 step 都遇到 prefill。目前它只在 data parallelism 下生效：`DPEngineCoreProc._should_throttle_prefills()` 由跨 rank 共享的 `step_counter` 驱动，而 base `EngineCore._should_throttle_prefills()` 无条件返回 `False`。在单 engine 上设置该选项静默变成 no-op 且无告警。motivating deployment 是单引擎使用 decode context parallelism（`--decode-context-parallel-size 8`）：DCP 是单引擎内的并行，现有 DP-only wiring 根本到不了它，尽管该 workload 与 DP 部署一样受益于这个 cadence。

## 优化点

在 scheduler 中、`data_parallel_size == 1` 时应用同样的 interval。没有 peers 就无需对齐，所以 interval 从"最后一个带 prefill 的 step"起算而非对齐全局 grid，因此不需要共享计数器；DP 部署不动，保留其 rank-aligned cadence。与现有 DP gate 一样，仅当某个 running request 能就地 decode（`any(not r.is_prefill_chunk for r in self.running)`）时才推迟 prefill，interval 不会让模型空转；默认 interval 为 `1` 时关闭该 gate，admission 行为不变。正文另说明两个相邻但不同的 PR：#49075 限制并发 prefill 数量、#53571 按 TTFT 目标重排 waiting queue，本 PR 只控制某个 step 是否 admit prefill 且不重排；#52969 是 aborted requests 的无关修复。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: 在 data_parallel_size==1 时按 last_prefill_step 实现 local prefill interval 与 never-idle 判断
vllm/config/scheduler.py: prefill_schedule_interval 配置
tests/v1/core/test_scheduler.py: 5 个新测试（holds_prefill_between_bursts、defaults_to_no_gating、left_to_engine_under_dp、never_idles_the_model 等）
tests/v1/core/utils.py: create_scheduler 增加 prefill_schedule_interval 参数
```

关键代码（来自 `vllm/v1/core/sched/scheduler.py` diff 末尾）：

```diff
+        # whether a new admission or a running chunk.
+        if self.local_prefill_interval > 1 and (prefill_scheduled or prefill_admitted):
+            self.last_prefill_step = self.num_executed_steps
+
         assert token_budget >= 0
         assert input_budget >= 0
```

## 性能声明

Kimi-K3：Kimi-K3 FP4、8x MI355X、single engine with DCP=8 与 LMCache KV offload、replay agentic-coding trace over 3600s measured window；Interactivity 为 p90 per-user output speed（越高越好）。测量配置同时启用本 PR 的 `prefill_schedule_interval=33` 与 #54625 的 `--cache-aware-admission-window 64`，baseline 为同 build 无这两选项，故数字为两选项合效而非本改动单独效果。p90 interactivity (tok/s/user)：concurrency 44 `11.94 → 17.28 (+44.7%)`、48 `11.32 → 16.33 (+44.3%)`；throughput 基本不变，`9,617 → 9,390 tok/s/GPU` at 44 (−2.4%)、`10,537 → 10,103` at 48 (−4.1%)。Workload 为 `semianalysis_cc_traces_weka_062126` agentic-coding trace，MiniMax-M3 agentic input mean 104k、p50/p90 77k/183k、output mean 687、~151:1，Kimi-K3 agentic input mean 134k、93k/266k、output mean 851、~158:1，约 150:1 prefill-dominated。MiniMax-M3：MXFP4 on MI355X、tensor parallel 4、prefix caching enabled、`offloading=none`、同 trace over 1200s，flags `--enable-chunked-prefill --max-num-batched-tokens 32768`、`--enable-prefix-caching`、fp8 KV cache、EAGLE3 spec decode，两选项 `--prefill-schedule-interval 25`、`--cache-aware-admission-window 64`、`--cache-aware-admission-threshold 0`；p90 interactivity concurrency 24 `130.43 → 140.72 (+7.9%)`、28 `143.47 → 165.90 (+15.6%)`、32 `122.37 → 129.15 (+5.5%)`、48 `101.83 → 107.97 (+6.0%)`，throughput 四点 `−3.2% 至 −0.1%`，prefix-cache 命中 81-90%（理论上限 96.2%）。正文注明 MiniMax-M3 数字取自该选项最没优势的配置，应读为下界，且两选项未按并发重调，数字为 floor。两模型均 No accuracy eval was run（不触及计算或采样，但 batch composition 变化使输出非 bit-identical）。

## 验证与回归风险

5 个新测试在 `tests/v1/core/test_scheduler.py`：`162 passed (157 pre-existing + 5 new)`；另 `test_async_scheduler.py` 等 `30 passed`、`test_priority_scheduler_random.py 12 passed`。环境说明：主机无 torch，测试在 `vllm/vllm-openai-rocm:nightly` 镜像（vllm 0.28.1rc1.dev108+g1dc464d42、torch 2.12.0、Python 3.12）内把本 checkout 源码覆盖到已安装包上跑，偏离 `AGENTS.md` 的 uv/.venv 流程，作者计划合并前按标准流程重跑；`tests/v1/core` 单进程整体跑会被 SIGKILL（main 上亦然），故按文件分别跑。风险面：单引擎（含 DCP）的 prefill admission 节奏、batch composition、与 LMCache 等 async KV connector 组合（正文提到 parked-in-lookup 使 `schedule()` 返回空、修前代码把这些调用计入 interval 的 failure mode）。

## 关联

#54625（cache-aware admission window，测量时与本 PR 选项合用）、#49075（max_num_partial_prefills，相邻不同问题且可合用、可能因合并顺序产生文本冲突）、#53571（SLO-aware scheduling，相邻不同问题）、#52969（throttle_prefills 的无关修复）。正文明确说明无 open PR 触及 `prefill_schedule_interval` 或 `_should_throttle_prefills`，均为正文出现的编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
