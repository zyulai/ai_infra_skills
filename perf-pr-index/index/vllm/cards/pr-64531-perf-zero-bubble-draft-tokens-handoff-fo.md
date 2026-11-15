# PR #64531: [Perf] Zero bubble draft tokens handoff for structured outputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yzong-rh |
| 合并 | 2026-11-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | speculative-decoding, ready, ci/build, mrv2, scheduler |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64531 |

## 一句话结论

`DraftTokensHandler` 从全局单一 copy barrier 改为每个 batch 一个同步 barrier:structured outputs + spec decode + async scheduling 下,scheduler 取 Batch A 的 draft tokens 不再被 Batch B 的 D2H copy 阻塞,pipeline bubble 消除;B200 PP=2 C64(紧约束)+2.68% E2E。

## 动机

structured output 请求需要 scheduler 侧拿到 worker 提议的 draft tokens 来计算 grammar bitmask 并推进 grammar;这由 `DraftTokensHandler` 异步地从 worker 侧拷贝。async scheduling 下 scheduler 领先 worker 并维护 batch queue(`batch_queue_size = PP + 1`),但 main 上的 handler 只维护一个 barrier(`_copy_pending` + 单个 event):Batch C 调 `get_draft_tokens` 取 Batch A 的 draft tokens 时,会与 Batch B 的 `set_draft_tokens` copy 竞争,Batch C 必须等 Batch B 的拷贝完成才能推进,尽管它只需要 Batch A 的——形成 pipeline bubble。PP=1 下少见(运行中请求通常连续调度),但 skip 也会造成 bubble(#66081);PP>1 下 decode 请求要隔 PP 步才能再调度(`Request.next_decode_eligible_step` 强制),skips 常见,故 E2E 收益只在 PP>1 部署上显著。

## 优化点

- 每个 batch 维护一个同步 barrier:新增 `_PendingCopy` dataclass(持有 `torch.cuda.Event`,同步一次后清空以跳过冗余 driver 调用)。
- `set_draft_tokens` 每次 copy record 一个 blocking(sleep)event(避免 busy-poll CUDA driver lock),并把 `(row, copy)` 存入 per-request 表,替代原来的单一 `_copy_pending` 标志。
- `get_draft_tokens` 只对实际要取的行调用 `copy.wait()`,不再无条件全局同步;无 structured-output spec decode 时仍按 draft width 返回占位符。

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/utils.py: 新增 _PendingCopy;DraftTokensHandler 按 batch 存 (row, copy),get_draft_tokens 只等待所需 copy
tests/v1/worker/test_draft_tokens_handler.py: 新增 grammar validation 只 fetch 最新 scheduler output 请求的断言
```

```python
@dataclass(slots=True)
class _PendingCopy:
    # Cleared once synchronized, to skip redundant driver calls.
    event: torch.cuda.Event | None

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()
            self.event = None
```

## 性能声明

- 声明:+3% E2E throughput at concurrency 64 vs main(`deepseek-ai/DeepSeek-V4-Flash-0731`,`dspark` spec decode,structured outputs 请求,async scheduling,PP=2,TP=1);sync scheduling 与 TP=2、PP=1 下无显著变化或回归。
- 实测(B200,`--num-repeats 10`,dspark 7 draft tokens + async scheduling + structured outputs;broad vs tight 约束、1 vs 64 并发):
  - Async,TP=1,PP=2:Broad C1 727.60±0.50 → 732.30±3.31 tok/s(+0.65%);Broad C64 10,991.95±101.22 → 11,105.82±121.56(+1.04%);Tight C1 728.02±0.53 → 735.48±3.68(+1.02%);Tight C64 10,964.57±125.17 → 11,258.04±177.77(+2.68%)。C64 两例 stat-sig。
  - Async,TP=2,PP=1:Broad C1 +0.65%;Broad C64 15,225.78 → 15,145.77(−0.53%);Tight C1 +1.63%;Tight C64 15,216.80 → 15,088.88(−0.84%)。无 stat-sig 变化。

## 验证与回归风险

- `VLLM_GPU_SYNC_CHECK=error` 下:`pytest tests/v1/worker/test_draft_tokens_handler.py tests/v1/e2e/spec_decode/test_distribution_shift.py tests/v1/e2e/general/test_async_scheduling.py` 9 passed;`tests/utils_/test_gpu_sync_debug.py` 22 passed;`tests/v1/e2e/spec_decode/draft_model/test_structured_output_pp.py` 1 passed。
- 新增 handler 测试:grammar validation 只 fetch 最新 scheduler output 中的请求;随后无 structured-output spec decode 时取全部 retained drafts。
- 风险:收益集中在 PP>1;TP=2/PP=1 高并发有小幅(无统计显著性)回退;per-batch event 带来少量对象分配;blocking event 语义保持不变(仍避免 busy-poll)。

## 关联

- #63219(本 PR 为其 follow-up);#66081(structured outputs + spec decode + async scheduling 的 bugfix,已合并,PP=1 skip 造成 bubble 的案例)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
