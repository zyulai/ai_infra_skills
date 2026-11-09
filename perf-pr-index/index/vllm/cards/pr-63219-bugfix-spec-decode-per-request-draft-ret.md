# PR #63219: [Bugfix][Spec Decode] Per-request draft retention for structured output under PP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yzong-rh |
| 合并 | 2026-10-30 |
| 分类 | Speculative Decoding \| tier: major |
| Label | bug, speculative-decoding, ready, ci/build, mrv2, scheduler |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/63219 |

## 一句话结论

按 request ID 保留 draft tokens，恢复 PP=2 结构化输出下 spec decode 吞吐（c96 最高 +444.02%）。

## 动机

正文说明本 PR revert #54442 并在 `DraftTokensHandler` 中直接解决根因，重新修复 #54437。#54442 通过 mask 掉 handoff race 之后的 draft tokens 修补 #54437，但缩短了 draft 序列，导致 PP + spec decode + structured outputs 下 acceptance-length（及性能）回归。本 PR 是 #61348 的 rework（rebase、处理剩余评论、修 pre-commit、简化实现与测试），在修复原问题的同时保持 acceptance length。

## 优化点

`DraftTokensHandler` 允许 worker 异步把 draft tokens 送回 scheduler / structured output manager。async scheduling 下 PP>=2 时，worker 连续 `set_draft_tokens(micro-batch A)`、`set_draft_tokens(micro-batch B)` 会让 B 覆盖 A，scheduler 无法恢复 A 的 drafts，生成错误 grammar bitmasks（#54437）。#54442 通过截断 worker 侧 draft 序列修复，代价是降低 acceptance length。本 PR 让 `DraftTokensHandler` 按 request ID 跟踪 draft tokens，使 micro-batch B 不能覆盖 micro-batch A。正文 Background 给出的 async scheduling 时序为：scheduler 侧 schedule step N、sample step N，worker 侧 execute step N，随后 scheduler 侧 schedule step N+1、wait for step N，worker 侧 sample step N 并 `set_draft_tokens`、execute step N+1，最后 scheduler 侧 `get_draft_tokens`、generate sampling bitmasks、sample step N+1。PP>=2 时 worker 可能在一次等待窗口内对 micro-batch A 与 B 连续 `set_draft_tokens`，B 覆盖 A 后 scheduler 便无法为 A 生成正确 bitmask。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: scheduler 侧按 request 取用 drafts 与 grammar bitmask
vllm/v1/core/sched/output.py: 调度输出携带 per-request draft 信息
vllm/v1/worker/gpu/model_runner.py: worker 侧按 request ID 设置 draft tokens
vllm/v1/worker/gpu/structured_outputs.py: 结构化输出使用对应 request 的 drafts
tests/v1/e2e/spec_decode/draft_model/test_structured_output_pp.py: 新增 PP + 结构化输出 E2E repro 与 acceptance 回归断言
```

## 性能声明

条件：`Qwen/Qwen3-4B` + `RedHatAI/Qwen3-4B-speculator.dspark` + 8 draft tokens + async scheduling + structured outputs，H100 80GB；命令 `.venv/bin/python run.py --output ... --pipeline-parallel-size ... --num-repeats 10`；吞吐单位 tokens/s，方括号为 95% 置信区间；broad vs tight 指 less / more vocab masked，c1 vs c96 指低 / 高并发。正文 "No regression on PP=1"：broad/c1 673.6 [668.3, 678.9] → 675.7 [666.5, 684.9]（+0.32%），broad/c96 9,835.7 [9,749.2, 9,922.2] → 9,785.8 [9,690.9, 9,880.7]（−0.51%），tight/c1 1,057.8 [1,047.7, 1,067.9] → 1,062.4 [1,051.7, 1,073.1]（+0.43%），tight/c96 18,582.1 [18,469.9, 18,694.3] → 18,565.3 [18,389.4, 18,741.2]（−0.09%）。"Recover performance regression on PP=2"：broad/c1 661.45 [660.82, 662.08] → 662.79 [662.39, 663.19]（+0.20%），broad/c96 3,951.12 [3,822.93, 4,079.30] → 12,038.09 [11,768.25, 12,307.93]（+204.68%），tight/c1 1,021.45 [1,020.70, 1,022.19] → 1,024.63 [1,023.00, 1,026.26]（+0.31%），tight/c96 3,865.53 [3,771.94, 3,959.13] → 21,029.32 [20,574.89, 21,483.75]（+444.02%）。

## 验证与回归风险

E2E repro `tests/v1/e2e/spec_decode/draft_model/test_structured_output_pp.py`（`Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash-b16` + PP=2，混合 constrained / unconstrained 请求）：main 上 `_assert_specdec_metrics` 失败，显示 acceptance rate 与 acceptance length 回归 (0.056471083070452156, 0.08341)；revert #54442 后 `_assert_outputs_are_structured` 失败（出现 unconstrained sampling）；本分支 "1 passed, 3 warning"。更多测试：scheduler / async scheduler / structured_output 共 571 passed, 2 xfailed, 58 warnings；spec decode / worker / e2e draft_model 共 140 passed, 3 skipped, 19 warnings。回归风险：PP=1 表格显示无回归；改动涉及 PP + spec decode + structured output 与 async scheduling 路径；正文建议按 commit 独立 review（`e3595db` revert #54442、`c67afdec` track draft tokens by request ID）。性能对照按 PP=1 与 PP=2 两组分别给出：PP=1 用于确认改动没有引入回归，PP=2 用于确认此前被 #54442 引入的吞吐回退已经恢复；结果以 tokens/s 表示，方括号内为 95% 置信区间，数据取自正文给出的 gist 链接，在 H100 80GB 上以 10 次重复运行取得，正文未注明精度信息。

## 关联

正文明确线索：revert #54442、re-fix issue #54437、为 #61348 的 rework。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
