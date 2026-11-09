# PR #60161: [KV Connector][NIXL] Register HiSparse host pool under single TP Rank

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @praveingk |
| 合并 | 2026-10-08 |
| 分类 | Overlap / Scheduler \| tier: major |
| Label | ready, kv-connector, hisparse |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/60161 |

## 一句话结论

HiSparse 共享 host pool 改由 TP rank0 独占注册/拉取，注册量降 8x、ITL p50 改善 10-12%。

## 动机

HiSparse 用 shared host pool 时所有本地 TP ranks 共享一个 /dev/shm mmap，但每个 rank 都把整个 pool 注册给 NIXL、并把相同 host-pool KV READ 到相同 pages。UCX 下等于每个 NIC 对全池做 ibv_reg_mr（relaxed ordering 时每 NIC 两次）；TP8 + 8 NICs 时每节点注册 128 次、每请求 KV 被拉 8 次。正文记载在 Cloud VPC（QEMU/KVM + ConnectX-7 SR-IOV VFs over RoCE、UCX 1.21 via NIXL 1.3.2）上，GLM-5.3 HiSparse decode 在 384 GiB pool 注册期间把两台 VM 都冻住、需重供节点，同样配置在裸机 H200 + InfiniBand 上通过。

## 优化点

新增 opt-in flag `VLLM_NIXL_HISPARSE_DRAM_RANK0_ONLY=1`（正文 Changes 提及 vllm/envs.py，默认 off），让 TP rank0 成为唯一注册共享 pool 并拉取的 rank；其它 ranks 保留 DRAM regions 于 metadata（prefill handshake 仍见相同 region 列表）、跳过 DRAM register_memory、跳过 DRAM transfer dlist、跳过每次 READ 的 DRAM 部分。GPU-resident 的 indexer cache 仍由每 rank 读、每 rank 仍通知 producer。`_leaves_dram_xfer_to_rank0()` 要求 flag、mixed DRAM/VRAM regions、tp_rank != 0 且 use_shared_hisparse_host_pool()，private pool 不受影响；pull_worker 的 `_read_blocks_mixed()` 在跳过 rank 上丢弃 DRAM descriptors、保留 device 部分，即便无内容可读仍注册请求以完成并通知。完成与排序无需新机制（请求仅在所有 TP ranks 报告 done 时完成、HiSparse 下一步从 shared pool restore、每 rank 的 cudaHostRegister pin 不变）。NixlPushConnector 已拒绝 mixed-memory KV caches，故不变。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py: _leaves_dram_xfer_to_rank0 决策、跳过 DRAM 注册/dlist、shutdown 忽略 None
vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py: _read_blocks_mixed 丢弃跳过 rank 的 DRAM 描述符
tests/v1/kv_connector/unit/test_nixl_connector.py: rank 决策、无 DRAM dlist、只读 device 与无描述符的 READ 仍通知
tests/v1/kv_connector/unit/test_nixl_connector_hma.py: 相关单测适配
```

```python
        if not (
            self._mixed_mem_types and self.kv_cache_config.hisparse_shared_host_pool
        ):
            return False
        skip = self.tp_rank != 0
        logger.info(
            "HiSparse host pool: NIXL DRAM registration and reads on TP rank 0 "
            "only (tp_rank=%d, skip=%s)",
            self.tp_rank,
            skip,
        )
        return skip
```

## 性能声明

注册 footprint（2-node decode DP2×TP8、vLLM 87ecf7dbb 上等价启动 patch）：100 GiB pool 由 136.1x pins/page / 12.5 TiB NIC-registered 降为 24.2x / 1.58 TiB；288 GiB 由 136x / 36.9 TiB 降为 24.05x / 4.51 TiB；384 GiB 由 136x / 49.2 TiB（两台 VM 冻住）降为 24.04x / 6.01 TiB（passes）；384 GiB 且 relaxed ordering off 为 16.02x / 3.01 TiB。正文称下降 8x 的是注册次数、pinned memory 与 NIC metadata，启动本身并未更快。P/D 正确性：18 / 28.7k / 250k-token prompts 在 100 / 288 / 384 GiB pool 下匹配 prefill-only reference，每请求 9 次 NIXL transfers（rank0 host + 8 device）而非 16，0 失败。E2E（4P/4D、64 H100、DP4×TP8、288 GiB shared pool、cc_traces 256k constant load、900 s/level）：c32 Out tok/s 602 vs 503（+19.7%）、TTFT p99 21.6 vs 89.0 s、ITL p50 20.9 vs 23.3 ms、ITL p99 116 vs 167 ms；c64 547 vs 563、c128 482 vs 520（两臂 prefill-bound）；"ITL improves at every level: p50 by 10–12%, p99 by 28–44%"。Transfer totals：NIXL bytes pulled 5.35 TB vs 31.3 TB、平均传输 30.8 vs 122 ms、p90 63.7 vs 318 ms、posting transfers 的 CPU 时间 472 s vs 3657 s。测试条件：模型 GLM-5.3、DeepEP LL decode，硬件 llmd-roce cluster / 8 NICs / RoCE VFs / ConnectX-7 / UCX 1.21 via NIXL 1.3.2、64 H100，relaxed ordering on（除注明外），workload cc_traces 256k。

## 验证与回归风险

单测覆盖：flag on/off × rank0/non-zero × shared/private pool 的 rank 决策、跳过 rank 无 DRAM dlist（descriptor 位置仍覆盖 DRAM regions 以保持 id 对齐）、只读 device 部分的 READ 仍通知、无描述符的 READ 仍完成并通知、既有 object.__new__ fixtures 初始化新属性；另有 P/D 输出匹配。回归风险：flag 默认 off（opt-in），仅 mixed DRAM/VRAM + shared host pool + tp_rank!=0 生效，private pool 不受影响；正文针对 Cloud RoCE VM 冻结与 384 GiB pool；注册次数降但启动不更快；NixlPushConnector 不变。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
