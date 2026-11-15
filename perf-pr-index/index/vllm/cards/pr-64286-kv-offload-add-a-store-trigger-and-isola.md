# PR #64286: [KV Offload] Add a store trigger and isolated store trigger handlers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amirfr3 |
| 合并 | 2026-11-12 |
| 分类 | Speculative Decoding(tier: major) |
| Label | ready, kv-connector, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64286 |

## 一句话结论

KV offload 引入 `StoreTrigger` 枚举(ON_COMPUTE / 新增 ON_HANDOFF)并为每个 trigger 配一个相互隔离的 store trigger handler,让 offloading manager 知晓每次 store 的原因;为 #61396 的下一阶段(ON_OVERWRITE、manager 自选 trigger、lazy store mode)铺路。

## 动机

#61396 的 Phase 1(设计讨论归 RFC):把每次 store 的"原因"交给 offloading manager,并把 connector 的 store 代码按 trigger 拆进各自 handler;为 KV cache manager 即将复用的 state block 增加 `ON_HANDOFF` 触发(该 store 不能推迟)。本 PR 取代 #61426。

## 优化点

1. 新 `StoreTrigger` 枚举:`ON_COMPUTE`(chunk 完成)与 `ON_HANDOFF`(state block 即将被复用,store 不可推迟)。
2. `prepare_store` / `complete_store` 增加必需的 keyword-only `trigger`,`complete_store` 另带 keyword-only `success`;out-of-tree manager 必须补这两个参数。
3. Connector 通过 `offloading/store_triggers/` 下每个 trigger 一个 `StoreTriggerHandler` 生成 store job;handler 经 `StoreJobSink` 提供 keys 与 jobs;job ID、状态、trigger 戳、fence、flush 与完成状态收归 scheduler。
4. Handler 完全隔离:互不感知、各自持有状态;scheduler 持有全局状态并给每个 handler 相同的 hooks(`on_load_scheduled`、`on_request_removed`、`on_request_restarted`、`reset`、`first_live_chunk_idx`、`build_store_jobs`)。
5. `first_live_chunk_idx` 返回 handler 仍可读其 GPU block ids 的组内第一个 chunk(无则 `None`);scheduler 只从最小值起清零 stale sliding-window block ids。
6. 配置含 copy-on-write source group(`MambaSpec` with `mamba_cache_mode="align"`)时,scheduler 在 manager 的 trigger 之前加挂 `OnHandoffStoreHandler`;job-ID 顺序不变,manager 可在 `prepare_store` 拒绝这些 store。
7. Tiering manager 把 trigger 传给 primary tier;promotion 不是 store,`prepare_write`/`complete_write` 是真实方法。
8. 合并 main 时顺带吸收 #65448:scheduler 保留 `num_hash_lookahead_tokens`;retention boundary `num_prompt_tokens - max(1, num_hash_lookahead_tokens)` 移入 `store_triggers/candidates.py` 的 `select_store_candidates`(仅 `OnComputeStoreHandler` 调用);`OffloadingConnector.supports_lookahead_block_hashes` 删除(基类默认已 True);`SchedulerOffloadConfig` 增加 `cache_hit_alignment_tokens` 与 `drop_eagle_checkpoint_block`,DCP-aware 的 `supports_partial_tail` 取代旧的 uniformity/DCP-width 检查。

## 关键改动

(注:pack 的 diff 截断,有 22 个文件未展示,已展示部分仅为 test_scheduler.py;下列实现文件清单来自正文描述。)

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/store_triggers/(新增): OnComputeStoreHandler / OnHandoffStoreHandler / candidates.py(select_store_candidates)
vllm/distributed/kv_transfer/kv_connector/v1/offloading/state.py(正文): RequestOffloadState + SchedulerOffloadConfig
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: handler 注册顺序 ON_HANDOFF → ON_COMPUTE;job 状态/trigger 戳/fence/flush 归 scheduler
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py: prepare_store/complete_store 增加 trigger kwarg;handler 隔离与顺序测试
```

```python
def test_handoff_handler_runs_before_the_selected_trigger():
    scheduler = _make_partial_tail_scheduler()
    assert list(scheduler._store_handlers) == [
        StoreTrigger.ON_HANDOFF,
        StoreTrigger.ON_COMPUTE,
    ]
    _make_partial_tail_request(scheduler)
    req_status = scheduler._req_status["req"]
    assert _on_handoff(scheduler).first_live_chunk_idx(req_status, 1) is None
```

## 性能声明

(perf claim 正则未检测到;以下为正文给出的端到端 sanity 数据——每模型一个 H100、0 server errors,在 head `8e0158e016`(早于 main merge,merged tip 重跑待办)测得;Qwen base `8551e3a13f`;镜像 `vllm/vllm-openai:nightly` `0.34.0rc1.dev206`)

- **Qwen3-8B**:`client.py 32x3188, 3 rounds` 两侧 store/load/hits 完全一致(head 与 base 均 store 29.82 GB、load 0.009 GB、hits 0.03% (R3));详细计数两侧相同:29.822 GB stored、9.4 MB loaded、64 external hits、205741 queries 中 100800 GPU prefix hits——工作集留在 GPU prefix cache,CPU tier 几乎不参与,但对比仍 like-for-like。`prefix_repetition 48x4096`:head req/s 48.97 / mean TTFT 162.8 ms / TPOT 17.95 ms vs base 48.51 / 169.4 / 18.06。
- **Falcon-H1-0.5B(align)**:`client.py 32x3332, 3 rounds`:store 4.52 GB、load 2.38 GB、hits 90.0% (R3)。handlers 为 `['ON_HANDOFF', 'ON_COMPUTE']`,CoW source group 1(`MambaSpec`);测量 wrapper 计数:ON_HANDOFF 432 offers / 352 keys / 352 jobs,ON_COMPUTE 370 offers / 618 keys / 337 jobs;`ON_COMPUTE` 跳过 `requires_cow_source` 组,Mamba state 只经 handoff 路径到 CPU tier。`prefix_repetition 32x4096`:head req/s 16.98 / TTFT 436.3 ms / TPOT 19.12 ms。

## 验证与回归风险

- `pre-commit run` 在 merge commit 上通过(含 ruff、mypy);每个 commit 带 `Signed-off-by`。
- 合并基线:main `f0787a9ab8`(2026-11-12,吸收 #58619、#67152、#67098、#65448)。
- Merged tip `c618ca51bf`:`test_scheduler.py` 519 passed, 12 failed;同目录其余文件(`--ignore=test_scheduler.py`)227 passed, 2 skipped;`tests/v1/kv_offload/cpu/test_manager.py` 86 passed。
- 12 个失败全部为 host 限制、在未改动 main 上同样失败:6 个 `test_cache_reset*` 需要 `/dev/shm`(macOS 没有);6 个 `test_lookahead_partial_*` 需要 Triton spec decode(standalone CPU model runner 拒绝)。更早 head `8e0158e016` 全量:1614 passed, 13 skipped, 17 failed,同为这两类 host 限制;`test_shared_offload_region.py`、`test_gpu_worker.py` 同因 deselect。
- 11 个 pre-merge commit 各自通过 offloading_connector 目录与 cpu test_manager 测试。
- 已知问题(正文明示):greedy 文本在 CPU restore 后不 bit-stable——Falcon head 匹配 24/32 rounds(无 offload 对照 32/32),Qwen head/base 各 31/32;2026-11-01 run(byte-identical offload counters)Falcon head 29/32、base 30/32。漂移位于 Mamba-align CPU-restore 路径,先于本 PR 存在,未 root-cause。
- 兼容性:`prepare_store(keys, req_context, *, trigger)` / `complete_store(..., *, trigger, success)` 会 break out-of-tree managers,是唯一 break;无用户可见配置变化;eager 行为不变(aligned-boundary 与 partial-tail store 保持相同 source、keys、job-ID 顺序,只是搬进 `OnHandoffStoreHandler`)。

## 关联

- #61396(Phase 1,设计 RFC)、#61426(被本 PR 取代)。
- 合并时吸收:#58619、#67152、#67098、#65448。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
