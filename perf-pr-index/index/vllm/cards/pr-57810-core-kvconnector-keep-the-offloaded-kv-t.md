# PR #57810: [Core][KVConnector] Keep the offloaded KV tier across pause/sleep

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aoshen02 |
| 合并 | 2026-11-08 |
| 分类 | KV Cache / Memory(major) |
| Label | documentation, frontend, multi-modality, kv-connector, rust, scheduler |
| Diff | +305/-66,24 files(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/57810 |

## 一句话结论

sleep/pause 新增 opt-out 保留外置 KV offload 层(CPU/SSD):RL rollout wake_up 后重放 311,744 tokens 的 p50 从 4.421s 降到 0.490s,external prefix-cache 命中 0→311,296。

## 动机

`pause`/`sleep` 的目的是把 GPU 内存交还,而外置 KV connector 层(CPU/SSD offload)是唯一能跨过这次交接存活的缓存,但 pause/sleep 级联会把它一并驱逐。对 time-share GPU 的 RL rollout(`sleep(1)` → train → `wake_up()`),这层缓存决定 resume 成本:没有它就得对每个 in-flight sample 重新 prefill。是否仍有效取决于调用者是否在暂停期间换权重——block 哈希不覆盖权重,引擎在 pause 时无法判断,故默认保持"evict",只加显式 opt-out(参数最终实现名为 `reset_connector`;PR 摘要与实测命令里出现过 `clear_connector_cache` 旧名,以合并的 diff/docs 为准)。

## 优化点

新参数 `reset_connector`(默认 True,所有入口默认行为不变)从 `LLM.sleep`/`release_kv_cache_memory` 贯穿 `pause_scheduler`/`_reset_caches(reset_connector=)`,同步到 Python/Rust client 与 `/pause`、`/sleep` HTTP 端点;Rust client 只在 opt-out 时发送该参数,新 frontend 仍可驱动旧 engine。评审修出两个安全点:1) in-process engine 在保留 connector cache 时先 step 直到 pending connector store 完成(store 还在读将被 sleep 丢弃的 KV cache);2) preemption flush 只清 store、放过 load——`_reset_caches` 抢占的请求会在保留层时于"同一步"内 resume 并携带 load,旧断言(preempted 请求只能有 store)在该场景崩溃(offloading/scheduler.py:1792)。fail-closed:connector 声明 `supports_retained_cache_on_pause`(默认 False,`OffloadingConnector` 为 True,`MultiConnector` 取子连接器合取),不支持的 connector(如 `MooncakeStoreConnector`)在 pause 前直接报错拒绝,而不是 wake_up 时崩溃。

## 关键改动

```text
vllm/v1/engine/core.py: sleep/pause_scheduler/release_kv_cache_memory 增加 reset_connector;保留 cache 时 drain pending stores
vllm/entrypoints/llm.py + vllm/entrypoints/serve/dev/sleep/api_router.py: Python API 与 HTTP 端点(/sleep、/release_kv_cache_memory)
rust/proto/control.proto + rust/src/server/src/routes/sleep.rs: gRPC/HTTP wire(SleepRequest.reset_connector,proto 0.6.0→0.7.0)
vllm/v1/engine/core_client.py + llm_engine.py: 各 engine client 的签名透传
tests/v1/kv_connector/unit/test_offloading_connector.py: sleep 保留/重置 offloaded blocks 的端到端断言
```

## 性能声明

2x GB200(server 用 head 节点 4 GPU,TP4),DeepSeek-V4.1-Flash,nightly `0.29.1rc1.dev422`,`OffloadingConnector`(kv_both,block_size 128,`cpu_bytes_to_use` 32 GiB),16 并发 completions(每请求约 19.5k tokens,共 311,744 tokens)→ `sleep level=1` → `wake_up` → 重放同样 16 prompts:

| 重放(after wake-up) | 默认(清层) | 保留层(opt-out) |
|---|---|---|
| external prefix-cache hits | 0 / 311,744 | 311,296 / 311,744 |
| kv_offload_load_bytes_total | 0 | 1.31 GB |
| kv_offload_store_bytes_total | 1.31 GB(全部重算) | 0 |
| p50 latency | 4.421 s | 0.490 s |

作者附注:latency 列仅供参考——两臂顺序跑在同一 server 且共享 prompts,第二臂起点更暖;client 的 prompt 生成按 word 计数,`--prompt-tokens 4096` 实际产出约 19.5k tokens,绝对延迟受此影响,但命中/字节计数不受;测量时挂载的 patch 还含 #57813 的 `pin_in_flight_chunks`(作者论证 32 GiB 层不会发生驱逐,pinning 无关)。

## 验证与回归风险

`tests/v1/core/test_scheduler.py` 177 passed;offloading_connector unit 467 passed;`test_engine_core.py -k "pause or sleep_forwards"` 10 passed;`cargo test -p vllm-server -- pause_route sleep_route` 5 passed。新增关键测试:`test_sleep_keeps_offloaded_blocks_unless_reset`(多进程/单进程两模式)、`test_pause_clear_connector_cache_opt_out`、`test_preempted_request_resumed_in_the_same_step`(在上一修订版失败)、Rust wire-format pin 测试。风险:sleep/pause 全链路签名变更与 vllm-proto 版本跳跃;Mooncake 等未适配 connector 在 opt-out 时收到显式错误;与 #52360 同类的 preemption×in-flight transfer 交互(store flush 未动,是默认"evict"的理由之一)。

## 关联

非重复:#57296(KV event payloads)、#56754(request draining)、#49789(graph/runtime state across weight reloads);#57813(`pin_in_flight_chunks`,测量环境带入);#52360(preemption vs in-flight transfers);follow-up:weight 更新路径(`finish_weight_update`)强制驱逐该层。本批同分类(KV Cache / Memory):无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
