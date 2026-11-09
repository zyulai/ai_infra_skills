# PR #47941: [EC Connector] P2P NIXL + CPU EC Connector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @omerpaz95 |
| 合并 | 2026-09-04 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, cpu, kv-connector, mrv2, rust |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 18 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/47941 |

## 一句话结论

让 ECCPUConnector 支持基于 NIXL 的 P2P EC 共享，CPU offload 的 encoder cache 可跨 vLLM 实例复用。

## 动机

PR 正文 Purpose 指出本 PR 扩展 `ECCPUConnector`，支持基于 NIXL 的 P2P EC（encoder cache）共享。原 PR #42998 被拆成两部分：第一部分 #47423 是独立的 CPU 侧 offloading EC connector，本 PR 是其扩展，允许 CPU-offloaded 的 EC cache 通过 NIXL 在 vLLM 实例之间共享。控制面为 ZMQ（consumer DEALER → producer ROUTER），承载 msgpack `XferReq` / `XferAck`。依赖 PR #42433（EC Transfer Params）与 PR #47423。

## 优化点

Producer scheduler 绑定 ZMQ ROUTER，单 router thread 接收 `XferReq`、校验 `mm_hash` 在 `_local_encodings`（否则 NACK）、pin CPU mmap 目标块并向 consumer 预注册块发异步 NIXL `WRITE`，每轮 `_sweep_completions` 轮询完成、解 pin 并回 `XferAck`；`_pending_save` → `_local_encodings` 两阶段记账避免在 `save_caches` 完成前把新块当可服务。Producer worker `save_caches` 与 Consumer worker `start_load_caches` 在专用 copy stream 上完成 `encoder_cache` ↔ `mmap[block_indices]` 拷贝并 CPU 同步。Consumer scheduler 用按 `(host, port)` 懒加载的 DEALER pool（ZMTP heartbeat + `EVENT_DISCONNECTED`），`ensure_cache_available` 为未见 `mm_hash` 分配块、发 `XferReq`、ack 前返回 `pending=True`；`_loaded` mmap cache 支持后续请求本地重拷（`_pending_reload`）免回源。失败路径：producer NACK（未知 mm_hash、compat hash 不匹配、已驱逐）→ consumer 释放块回退本地 encode；producer 断连 → `HEARTBEAT_TIMEOUT`（4s）内 `on_peer_down` 驱逐 peer、移除 NIXL agent 并 tombstone 在途条目。配置：producer 必需 `--mm-encoder-only`，`num_ec_blocks` 默认 80000，consumer 无需静态 peer 配置、全靠 `ec_transfer_params` 按请求传递。

## 关键改动

```text
vllm/distributed/ec_transfer/ec_connector/cpu/connector.py: 增加 V2 model runner 校验与 request_finished 转发，把 ec_transfer_params 经响应体发布
vllm/distributed/ec_transfer/ec_connector/cpu/control/base.py、control/zmq.py: 新增 ControlConnection 抽象与 ZMQ ROUTER/DEALER 非阻塞实现
vllm/distributed/ec_transfer/ec_connector/cpu/data/base.py: 新增共享 mmap / 传输元数据结构
tests/v1/ec_connector/unit/test_scheduler_nixl_producer.py、test_scheduler_nixl_consumer.py、test_scheduler_nixl_ctor.py: producer/consumer 分支单测
tests/v1/ec_connector/unit/cpu/scheduler/test_embedding_cache.py: 新增 discard / pin_if_ready 行为测试
```

```diff
+    def request_finished(
+        self, request: "Request"
+    ) -> tuple[bool, "dict[str, Any] | None"]:
+        assert self.connector_scheduler is not None
+        return self.connector_scheduler.request_finished(request)
+
     def get_finished(
```

## 性能声明

PR 正文未给出定量性能数据（功能/架构 PR，无 benchmark 数字），也未给出测试条件，模型、硬件、精度与 workload 均 PR 未注明测试条件。

## 验证与回归风险

Testing plan 列出 `tests/v1/ec_connector/unit/` 下的单测：`test_metadata.py`（msgpack round-trip、compat hash factors）、`test_scheduler.py`（~10 tests，覆盖 alloc/send、幂等、ack 成功/失败、peer-pool 失效、router-thread WRITE/NACK、并发 xfers、`request_finished` shape）、`test_worker.py`（3 tests for save/load copy paths）、`test_ec_shared_region.py`、`test_utils.py`、`test_common.py`；命令 `python -m pytest tests/v1/ec_connector/unit/ -v`。CodeRabbit summary 提到新增 NIXL 远程传输、transfer coordination、compatibility checks、retry handling、peer connection management，以及 CPU encoder connector 现在经 scheduler 上报完成请求、V2 model runner 关闭时给出更清晰校验。回归风险：正文明确 `ECCPUConnector requires the V2 model runner`（否则抛 ValueError）；NIXL WRITE、ZMQ 心跳与 mm_hash 兼容校验引入跨实例依赖，PR 未注明更多受影响路径或硬件。

## 关联

正文明确：原 PR #42998 被拆分；第一部分 PR #47423（standalone CPU-based offloading EC connector）；依赖 PR #42433（EC Transfer Params）与 PR #47423。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
