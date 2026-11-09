# PR #52921: [Bugfix] Align CPU offload pool size across PP/spec-decode workers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @charxwu |
| 合并 | 2026-10-22 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, ready, kv-connector, verified |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/52921 |

## 一句话结论

修复 PP/spec-decode 下 CPU offload 池块数在 scheduler 与 worker 间不一致导致的越界 offload 拷贝:worker 间 all-reduce MIN 对齐。

## 动机

不均匀 pipeline parallelism 或 speculative decoding 下,较重的 PP rank 在同样字节预算内能存的 block 更少,而 scheduler 此前按较轻 rank 估算池大小,产生的 block ID 可能超过某个 worker 的分配,导致拷贝操作崩溃。社区此前需要 per-topology 手工调参(如 InferenceX 的 `KV_OFFLOAD_MAX_CPU_BLOCKS` workaround)。

## 优化点

三层对齐:worker 在分配 offload 存储前用 `sync_num_offload_blocks_across_workers` 在 world group 上做 all-reduce MIN;engine 仅在启用 SimpleCPUOffloadConnector(包括嵌在 MultiConnector 内的实例)且开启 prefix caching 时,经 collective_rpc 查询对齐后的块数,缺失结果视为初始化错误(fail closed)而非回退估算;Scheduler 直接使用该计数,删除 config 估算回退与 per-worker config 传递。显式 `num_cpu_blocks` 覆盖仍受支持并校验上限。`CacheConfig` 新增 `simple_cpu_offload_num_blocks` 且不计入 compute_hash(运行时量)。

## 关键改动

```text
vllm/v1/simple_kv_offload/sizing.py(新): sync_num_offload_blocks_across_workers all-reduce MIN
vllm/v1/engine/core.py: _initialize_kv_caches 查询 worker 对齐块数并断言 workers must report
vllm/v1/executor/abstract.py: get_simple_cpu_offload_num_blocks 取各 rank 最小值,不一致时告警
vllm/v1/simple_kv_offload/manager.py: _derive_cpu_config 优先 aligned_num_cpu_blocks
vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py: 递归查找嵌套 connector 与 get_aligned_num_cpu_blocks
```

## 性能声明

无吞吐/延迟数据。ROCm smoke(假权重,nightly 镜像):Kimi-K3 TP4×PP2 worker 50/46、scheduler 50 → 全部对齐 46,两遍生成结果一致;Kimi-K3 TP8×PP2 双节点 MI355X worker 809/746、scheduler 809 → 746,健康与完成 smoke 通过。

## 验证与回归风险

Python 3.12/CPU 测试 144 passed / 18 skipped(offload worker 模块与 17 个 GPU 依赖 embed-sharing 用例跳过,含 3 个 worker 同步测试,AMD GPU 用例留待 CI);6 个针对性回归覆盖 RPC gating、MultiConnector 与缺失 worker 结果(修复前该用例失败);Nixl CI 脚本附带 #62879 的 prefiller-ready 等待修复。风险:对齐取 MIN 会使部分 rank 的 offload 池变小(显存换正确性);GPU/PP smoke 未在最终 review 修复后重跑。

## 关联

同分类 Speculative Decoding:#43091、#50272;KV Connector 区域:#28309、#51052。依赖:#62869(main 合入)、cherry-pick #62879 的 11 行测试脚本改动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
