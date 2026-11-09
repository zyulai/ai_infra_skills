# PR #19330: [V1] [P/D] Add Support for KV Load Failure Recovery

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sdavidbd |
| 合并 | 2025-09-30 |
| 分类 | Other(major) |
| Label | documentation, ready, v1, kv-connector |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用但被截断(7 个文件未包含) |
| 链接 | https://github.com/vllm-project/vllm/pull/19330 |

## 一句话结论

为 KV connector 路径增加 KV 加载失败恢复:检测失败 block,自动从有效前缀重调度受影响请求重算,避免外部 KV offload/传输故障导致请求失败。

## 动机

PD 分离与外部 KV cache offload 场景下,connector 从外部系统加载 KV block 可能失败(连接问题、eviction 等),此前没有统一恢复机制,受影响请求无法继续。本 PR 实现 RFC #19329 的设计,同时支持同步与异步两种加载方式。

## 优化点

Connector 侧:KVConnectorBase 新增 `get_block_ids_with_load_errors()` 接口(同步加载在检测到的 forward pass 报告;异步加载最迟在 `get_finished()` 返回该请求的同一 pass 报告),`MultiConnector` 聚合多个子 connector 的失败 block。调度器侧:`KVConnectorOutput` 增加 `invalid_block_ids`;`update_from_output` 中调用 `_handle_invalid_blocks`,对 waiting(异步,WAITING_FOR_REMOTE_KVS)与 running(同步)请求扫描失败 block,把 `num_computed_tokens` 截断到首个失败 block 之前,受影响请求重调度重算;共享 block 只由第一个请求重算(目前仅 sync 路径支持共享),`failed_recving_kv_req_ids` 记录异步失败请求,加载完成时仅缓存有效 computed tokens 或释放块等待本地缓存命中。Worker 侧:sync 失败时对齐 `output_token_ids` 与 input_batch 的 token 计数。另修复 `cache_blocks`/`cache_full_blocks` 在 `num_cached_blocks >= num_full_blocks` 时的重复缓存问题。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: _handle_invalid_blocks / _update_requests_with_invalid_blocks 重算恢复
vllm/distributed/kv_transfer/kv_connector/v1/base.py: get_block_ids_with_load_errors 接口
vllm/v1/worker/gpu_model_runner.py: sync 失败后回滚 output_token_ids 与 batch token 计数
examples/offline_inference/kv_load_failure_recovery/: 故障注入 connector 端到端示例
```

## 性能声明

PR 未给出性能对比数字;这是健壮性改进。验证结果为正确性:故障注入端到端示例中,模拟失败场景的所有输出与基线 decode 一致("Outputs match: recovery successful"),日志显示成功恢复(如 "Recovered from KV load failure: 2 request(s) rescheduled (1022 tokens affected)")。

## 验证与回归风险

单元测试 `tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py`(async 3 例、sync 3 例、shared blocks 3 例、async progressive 2 例)与 `test_output_aggreagator.py`;端到端示例用 `RogueSharedStorageConnector` 注入故障并 diff 输出。回归风险:触及调度器热路径 `update_from_output`(新增 invalid_block_ids 检查)与 `_update_states`;`CachedRequestData` 新增 `num_output_tokens` 字段(序列化格式变化);hybrid memory allocator 暂不支持(TODO);异步加载的 block 共享不支持;`get_num_new_matched_tokens` 文档新增"只计入实际可加载的前缀"的契约要求,connector 实现需遵守。

## 关联

实现 RFC #19329;同区域为 KV connector / PD 分离健壮性;pack 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
