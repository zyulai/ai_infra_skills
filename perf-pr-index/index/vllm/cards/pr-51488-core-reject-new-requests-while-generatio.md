# PR #51488: [Core] Reject new requests while generation is paused

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-10-19 |
| 分类 | Other（tier: major） |
| Label | documentation, frontend, ready, rust, scheduler, pooling |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「14 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/51488 |

## 一句话结论

暂停期间由 engine 拒绝新请求并返回 503，修复 resume 后用新权重服务旧请求的问题。

## 动机

正文说明 `pause_generation(mode="abort"|"wait")` 与 `sleep()` 把暂停当作 generation boundary，但暂停之后到达的请求会被静默排入已暂停的 engine，并在 resume 之后、在**新权重**下被服务，属于不正确行为。本 PR 解决 #51476 中的第二个问题，并按 @hao-aaron 的评审重新设计：以 engine（而非每个 frontend）作为 pause admission 的唯一事实来源。正文注明构建在已合并的 #59559 之上。契约是：`pause_generation(abort|wait)` 返回后到 `resume_generation()` 返回前，共享该 engine 的所有 API server 的新请求一律以 `EnginePausedError`（HTTP 503，可重试）失败，`keep` 仍接受并排队；与 pause 竞态的请求可被拒绝或视为 in-flight，但绝不挂起、不留残留。

## 优化点

Engine admission：`_reject_add_while_paused` 与 `_reject_add_in_shutdown` 并列、位于 `add_request` 之前，只在 `PAUSED_NEW`（abort/wait）拒绝，放行 `abort_immediately` 标记以保证 P/D KV cleanup 仍可执行；拒绝同时结束该 id 下打开的 streaming session，busy loop 串行处理 ADD 与 pause/resume 故无需锁。Surfacing：新增 `FinishReason.PAUSED`，`OutputProcessor` 把被拒 child 当正常完成处理并抛 `EnginePausedError`（`GracefulHTTPError`，503），`generate` 在 fan-out 完整发出后才 abort 请求并回收 `n>1` siblings；Rust frontend 在 `/v1/completions`、`/v1/chat/completions`、`/inference/v1/generate` 映射为 503（`engine_paused`）、gRPC 映射为 `UNAVAILABLE`。被拒的 streaming-input session 停止接收输入、其输入 task 被取消，DP 下 abort 沿 main 的 `streaming_reqs` 路由（#58942）；被拒 pooling 请求由 `AsyncLLM.encode` 抛 `EnginePausedError`，同步 `LLMEngine` over multiprocessing 得到空结果。DP 方面 #59559 已移除 frontend wake、由 engine 独自起 wave，故本 PR 无需 DP 专用改动；旧版的 client-side admission flag、control-plane lock 与 reopen 逻辑被删除。scheduler 未改动，`ABORT` 处理与 sleep/resume 同 main。

## 关键改动

```text
vllm/v1/engine/core.py: 新增 _reject_add_while_paused、pause_generation/resume_scheduler 与 reject_new_requests 标志，_send_finish_outputs_to_client 携带 error
vllm/outputs.py: 新增 RequestError 与 FinishReason.PAUSED（diff 中改动）
vllm/v1/engine/output_processor.py、async_llm.py、core_client.py: 被拒请求的完成、abort 与回收路径
rust/src/engine-core-client/src/client.rs: pause_scheduler→pause_generation 并声明拒绝语义
rust/src/engine-core-client/src/protocol/output.rs: 新增 RequestError 协议结构（含 retryable）
```

```python
    def pause_generation(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        future = self.pause_scheduler(mode, clear_cache)
        self.reject_new_requests = mode != "keep"
        return future

    def resume_scheduler(self) -> None:
        self.reject_new_requests = False
        super().resume_scheduler()
```

## 性能声明

PR 正文未给出定量性能数据（pack 的 perf claims 检测为 none）；正文在 Essential Elements 中明确写 "Control-plane and scheduling only, no effect on model output; no eval run"，即仅控制面与调度改动、对模型输出无影响、未跑 eval。PR 未注明测试条件（无性能基准）。正文给出的是各测试套件的通过数而非吞吐/延迟数字，本卡无法给出性能结论。

## 验证与回归风险

Test Result（全部运行于 `vllm/vllm-openai:nightly`（af7f9488c）、PR 文件 bind-mount，模型 Qwen/Qwen3-0.6B 与 ibm-research/PowerMoE-3b，硬件 H200；GPU 与另一任务共享，内存不足导致 engine 启动失败的用例单独重跑）：新单元测试 14 passed；`test_async_llm.py` pause 测试 10 passed；HTTP multi-frontend + RLHF pause lifecycle 7 passed；`test_scheduler.py` / `test_async_scheduler.py` / `test_output_processor.py` / `admission_control` / `streaming_input/` 306 passed（另 9 个不带本 PR 也失败或需 >1 GPU）；完整 `test_async_llm.py` + streaming-input e2e + RLHF 55 passed；`test_async_llm_dp.py` 28 passed / 4 skipped；DP=2 MoE、`--api-server-count 2` 六轮 pause/resume（abort/wait/keep）后 sleep/wake 均"rejected while paused, all served after resume"；Rust `vllm-engine-core-client` / `vllm-server` 132 与 469 passed；mutation checks 逐项回退验证测试会失败。Known limitations 列出多条边界：`wait` 对暂停前已准入请求仍按 main 行为、`stream=true` 以 SSE error chunk 收 503、同步 `LLMEngine` over multiprocessing 返回 `finish_reason="paused"` 而非抛错、`VLLM_ENABLE_V1_MULTIPROCESSING=0` 的 in-process engine 仍排队、`wait` + `clear_cache=True` 在 idle streaming session 持 KV 时缓存重置失败。可能影响路径：所有 API server 的准入、Python/Rust frontend 的 503 映射、streaming/pooling、DP wave 与 RLHF weight update 流程。

## 关联

正文明确引用：#51476（本 PR 解决其第二个问题）、#59559（已合并，移除 frontend wake，本 PR 构建其上）、#58942（DP 下 `streaming_reqs` 路由，abort 沿此路径）、#51487（保守版对应 PR）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
