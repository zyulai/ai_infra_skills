# PR #63827: [Bugfix][Core] Defer async KV loads on HiSparse decoders until in-flight steps finish

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vMaroon |
| 合并 | 2026-10-28 |
| 分类 | Other（tier: major） |
| Label | bug, ready, kv-connector, scheduler, kv-cache-manager, hisparse |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/63827 |

## 一句话结论

修复 HiSparse P/D decoder 上 NIXL 异步 KV load 覆写仍被 in-flight step 读取的 GPU page 导致的输出坍塌（正确性修复）。

## 动机

四个条件同时满足即触发 KV 损坏（fixes #63811）：HiSparse 开启时会把 running 请求的干净 resident pages 提前归还 GPU pool，但请求会继续读这些 page 直到 pool 重新分配它们；async scheduling（`max_concurrent_batches > 1`）使 step N 仍在 GPU 上时 step N+1 已启动；KV 经异步 load 到达（NIXL 从 CPU 发起读，没有任何机制把它的写入排在 step N 之后）；step N+1 恰好把其中一个归还的 page 分给新请求、而新请求的 KV 正来自该 load。于是 NIXL 在 step N 仍把它当作原 owner KV 读取时写入新 KV，owner 从 step N 起读到错误 KV，后续 step 又 attend 到这些 token，输出从此不恢复——表现为输出中段陷入单 token 重复（GLM-5.3 上 `locklocklock...` 直到 max_tokens），MTP acceptance 降为 0。

## 优化点

核心机制是"延迟一步注册 load"：当 `defer_async_loads` 生效（`defer_block_free` 开启且 KV cache config 含 HiSparse host pool）且当前有 step in flight（`sched_step_seq > processed_step_seq`）时，scheduler 不再在分配时立即调 `connector.update_state_after_alloc`，而是把 load 放入 `deferred_loads`（记录 fence = `sched_step_seq`），请求保持 `WAITING_FOR_REMOTE_KVS`。`update_from_output` 中 `_drain_deferred_loads` 在 `processed_step_seq` 达到 fence 后启动 load，与既有 `_drain_deferred_frees` 并排；connector prefix-cache stats（`hits_by_source`）随启动时补记。deferred 期间 abort 的请求未发出 load，由 scheduler 自己按同一 fence 延迟释放块。代价：HiSparse decoder 上每个异步加载请求晚约一步开始，无 device sync、无 page 复用检查。仅 HiSparse + `defer_block_free` 时激活，其他引擎行为不变。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: defer_async_loads 判定 + deferred_loads 队列 + _drain_deferred_loads
tests/v1/core/test_deferred_block_free.py: 4 个新单测（deferred 注册时机/hit 来源记录/无 defer 行为/abort 释放）
```

```python
self.defer_async_loads = (
    self.defer_block_free
    and kv_cache_config.hisparse_host_num_blocks is not None
)
```

## 性能声明

本 PR 为正确性修复，无吞吐类性能声明。验证数据（GLM-5.3，H100 P/D，decoder TP8 x DP2、MTP 3、MRV2、async scheduling、`MultiConnector`=Nixl+HiSparse、20,000-block pool；`benchmarks/multi_turn` 流量，46-77K token prompts、2-4K token answers）：

- 用一行测试改动（`append_n`→`prepend_n`，归还页优先复用）放大触发条件后：base 引擎 162 个答案中 90 次 page 在被读时交给 NIXL load、81 个答案坍塌（<1.3 token/step 持续 ≥512 token，坍塌起点中值为覆写后 35 token）；本 PR 173 个答案、106 次 load 被延迟持有、0 坍塌。
- 默认 free-queue 顺序下 base 为 133 答案 / 0 覆写 / 0 坍塌；长答案（12-16K）默认顺序、KV pool 最高 88% 占满时，base 的 5,215 次归还页复用中 0 次落入 NIXL 填充组——即默认分配顺序下该 hold 几乎不会触发。
- 代价：stress run 中后继轮次 TTFT p50 1.98 s（base）→2.07 s（本 PR）。

## 验证与回归风险

- 新增 4 个单测覆盖：deferred load 在 in-flight step 处理后才注册、启动时记录 `hits_by_source`、关闭 defer 时立即注册、deferred 期间 abort 后按 fence 释放块。
- 回归风险：只影响 HiSparse + async KV consumer + 重叠 step 的组合；延迟注册改变 `WAITING_FOR_REMOTE_KVS` 的停留时长（约多一步）；`finish_requests` 对 deferred load 的释放路径改变（经 `deferred_loads.pop`）；PR 自述 CI 不覆盖 GPU 路径（需 NIXL P/D on GPUs），验证依赖人工复现流程；GLM-5.3 数字在早期修订版（仅延迟进入复用页的 load）上测得，最终版为全部延迟。

## 关联

- 修复 issue：#63811；修订建议来自 LucasWilkinson（fork PR vMaroon/vllm#2）。
- 同区域：本批 kv-connector/scheduler PR #60695（Mooncake partial tail）、PR #44594（watermark 准入预留）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
