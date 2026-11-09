# PR #63219: [Bugfix][Spec Decode] Per-request draft retention for structured output under PP

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63219](https://github.com/vllm-project/vllm/pull/63219) |
| 作者 | yzong-rh |
| 合并日期 | 2026-10-30 |
| labels | bug, speculative-decoding, ready, ci/build, mrv2, scheduler |
| 热度 | 👍 1 · 💬 39 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
PP≥2 + 投机解码 + 结构化输出时微批次草稿互相覆盖导致约束失效;改为按请求 ID 保留草稿,并回退降低接受长度的旧补丁。

## 问题现象
在 pipeline parallelism(PP≥2)配合 speculative decoding 与 structured outputs(async scheduling)的场景下,结构化输出请求会出现不受语法约束的采样(issue #54437 报告的问题)。此外 main 分支上此前的补丁 #54442 引入了接受率/接受长度(acceptance rate / acceptance length)的性能回归:E2E 测试 `_assert_specdec_metrics` 失败,实测 `(0.056471083070452156, 0.08341)`;若直接回退 #54442,则 `_assert_outputs_are_structured` 失败,输出如 `["blue", "` 不再符合 schema。PP=2 高并发下吞吐严重退化:broad/c96 为 3,951.12 tokens/s,tight/c96 为 3,865.53 tokens/s(修复后分别提升 +204.68% / +444.02%)。

## 根因
PR 正文披露:`DraftTokensHandler` 允许 worker 异步地把 draft tokens 送回 scheduler / structured output manager。它只保存"最新一批"由 worker set 的草稿。PP=1 时没问题,但 PP≥2 时会出现微批次覆盖时序:

```
                        set_draft_tokens(micro-batch A)
                        set_draft_tokens(micro-batch B)
get_draft_tokens
```

微批次 B 覆盖了 A,scheduler 侧拿不到 A 的草稿,生成错误的 grammar bitmask,导致 #54437 的无约束采样。而 #54442 的修法是通过截断 worker 侧草稿序列来规避(handoff race 后 mask 掉 draft tokens),等效缩短草稿序列,造成接受长度下降的性能回归。

## 修复方案
回退 #54442,直接修 `DraftTokensHandler` 的 handoff race:按 request ID 追踪草稿,使微批次 B 无法覆盖微批次 A(`vllm/v1/worker/gpu/spec_decode/utils.py`):

```diff
-        self.req_ids: list[str] = []
-        self.draft_tokens_np: np.ndarray | None = None
+        self._draft_tokens_by_req: dict[str, np.ndarray | int] = {}
+        self._deferred_req_ids: Iterable[str] | None = None
+
+    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
+        self._deferred_req_ids = (
+            scheduler_output.scheduled_spec_decode_tokens.keys()
+            if scheduler_output.pending_structured_output_tokens
+            else None
+        )
```

配套改动:
- `set_draft_tokens` 把每个请求的草稿行写入 `_draft_tokens_by_req`(含无结构化输出请求时仅记录草稿宽度的分支);新增 `discard(req_id)`,`get_draft_tokens` 按需返回延迟请求或全部存活请求的草稿。
- `gpu_model_runner._remove_request` 中调用 `draft_tokens_handler.discard(req_id)`;`execute_model` 中调用 `begin_step(scheduler_output)`。
- 移除 #54442 引入的屏蔽机制:`GrammarOutput.num_acceptable_drafts` 字段、`grammar_invalid_drafts()`、rejection sampler 的 `invalid_drafts`/`verify_draft_sampled` 参数及相关 `stale_drafts` 逻辑。
- 新增 E2E 复现测试 `tests/v1/e2e/spec_decode/draft_model/test_structured_output_pp.py`(Qwen/Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16,PP=2,混合约束/非约束请求)和单元测试 `tests/v1/worker/test_draft_tokens_handler.py`,删除 `test_grammar_invalid_drafts.py`。

## 影响与回归风险
- 影响窗口:正文未明确给出版本/提交范围,但涉及 #54442(引入回归的补丁)到本 PR 之间的 main 分支。
- 回归面:改动集中在 speculative decode + structured outputs + async scheduling 路径:`DraftTokensHandler` 的状态管理、`GrammarOutput` 字段签名变化(影响 `scheduler.get_grammar_bitmask`、`batch_shard._shard_grammar_output`)、rejection sampler 的 `_verify`/`_verify_in_chunks`/`__call__` 签名变化,以及流式输入测试对 `draft_tokens_handler` 的 mock。PP=1 性能数据与 main 持平(±0.51% 内);所有列出测试通过(571 passed / 140 passed 等)。

## 关联
- https://github.com/vllm-project/vllm/pull/54442
- https://github.com/vllm-project/vllm/issues/54437
- https://github.com/vllm-project/vllm/pull/61348
- 性能数据: https://gist.github.com/yzong-rh/e15dc602fbfd11e2b64c2d41527a1a72

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
