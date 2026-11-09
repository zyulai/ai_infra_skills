# PR #35264: [KV Connector]: Support KV push from Prefill to Decode node using Nixl KV Connector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @snadampal |
| 合并 | 2026-06-12 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | documentation, ready, v1, kv-connector |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾标注 21 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/35264 |

## 一句话结论

Nixl KV Connector 支持 P→D 主动 push，1p1d 下 TTFT 相比 pull 模式改善 1.2x-3.0x。

## 动机

PR 正文 Purpose：改善 P-D disaggregated inference 部署的 TTFT。正文说明实现了 KV push 特性——Prefill 节点在 model executor 完成 forward pass 并结束请求后，立即把 KV blocks 推送到 Decode 节点（对比原有的 pull 模式）。正文开头引用 RFC issue #36923。

## 优化点

把 KV 传输从 Decode 侧拉取改为 Prefill 侧在请求完成时主动 push，以缩短 TTFT。实现支持 P 与 D 节点间的 heterogeneous TP 与 heterogeneous block sizes，并覆盖两种时序：Scenario 1（D 在 P 生成完 KV 之前就向 P 注册 blocks）与 Scenario 2（P 在 D 注册前已备好 KV）。`factory.py` 注册 `NixlPullConnector` 与 `NixlPushConnector`；`base.py` 新增 `has_pending_push_work()` 钩子，使 P 侧仍有待写入 D 节点的 KV blocks 时 engine 主循环继续 stepping（注释写明不实现 push 的 connector 保持 False）；`multi_connector.py` 用 `any(...)` 聚合该标志；`nixl` 包拆出 `base_scheduler.py` / `base_worker.py` 共享基类，并在 `__init__.py` 引入 pull/push 各自的 scheduler 与 worker。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/factory.py: 注册 NixlPullConnector / NixlPushConnector
vllm/distributed/kv_transfer/kv_connector/v1/base.py: 新增 has_pending_push_work() 钩子
vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py: 聚合 has_pending_push_work
vllm/distributed/kv_transfer/kv_connector/v1/nixl/__init__.py: 引入 base/pull/push scheduler 与 worker
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py、base_worker.py: 拆分共享基类
```

```text
+KVConnectorFactory.register_connector(
+    "NixlPullConnector", "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
+    "NixlPullConnector",
+)
+KVConnectorFactory.register_connector(
+    "NixlPushConnector", "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
+    "NixlPushConnector",
+)
+    def has_pending_push_work(self) -> bool:
+        return False
```

## 性能声明

正文：`vllm bench serve` 使用 sonnet dataset、不同 input/output token lengths，"KV push mode (this PR) showed 1.2x - 3.0x improvements over pull mode across different input and out lengths"，性能数字测于 AWS P5en instance，并称在 AWS Trn2 instances 上观察到相近改善（该特性最初在 Trn2 上开发）。表格（Prefill/Decode TP、Input Len、Output Len、QPS、Mean/Median/P99 TTFT、Mean TPOT、Req/s）示例：TP4 512/64 QPS4 pull → push Mean TTFT 110.94 → 74.46 ms、P99 740.03 → 100.24 ms；TP4 512/128 QPS4 301.92 → 93.77 ms、P99 1125.98 → 302.37 ms；TP4 512/128 QPS8 2948.11 → 1212.48 ms、P99 7258.96 → 2826.72 ms、Req/s 4.65 → 5.92；TP8 512/128 QPS16 3672.89 → 1653.70 ms、Req/s 6.45 → 9.04；TP8 2048/128 QPS8 1048.93 → 648.08 ms。各组 Mean TPOT 基本不变（约 8.84-12.88 ms）。

## 验证与回归风险

Test Plan："Manually tested Inference on P-D disaggregated setup"；Test Result："Tested 1p1d configuration for pushing KV from prefill to decode node and validated accuracy of the results"。正文 TODO 列出 "1. Run unit tests"（即单测尚未运行），Phase2 列出 "Add layer-wise KV push to bring out the actual performance gains from kv push model"；Checklist 四项均未勾选。回归风险：正文声明支持 heterogeneous TP / block sizes 与两种注册时序场景，但未给出单测结果或其他 KV connector、精度路径的回归说明。

## 关联

无（正文开头引用 RFC issue #36923，非 PR 编号；TODO/Phase2 未给出 PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
