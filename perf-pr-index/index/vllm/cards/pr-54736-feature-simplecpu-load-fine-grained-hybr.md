# PR #54736: [Feature][SimpleCPU] Load fine-grained hybrid prefix hits

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @YukioZzz |
| 合并 | 2026-09-11 |
| 分类 | Other (tier: major) |
| Label | ready, kv-connector, verified, kv-cache-manager |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(2 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/54736 |

## 一句话结论

让 SimpleCPU offload 在混合模型(Mamba+attention)上可靠服务细粒度外部 prefix 命中:replay 时外部命中率从 0 提升到 86–99%,输出与 cold 完全一致。

## 动机

混合架构模型上 SimpleCPU offload 无法可靠地服务细粒度外部 prefix 命中,根源有二:其一,Mamba `align` 模式会把内部状态置 null 并原地重定位 speculative block,其 block table 无法用单调递增的位置游标解析,连接器必须直接消费 KV cache manager 已发布的精确边界交接(boundary handoffs);其二,prompt 可能在 hash 边界结束而该边界并非 full-attention 的 effective-block 边界,纯位置扫描只保留完整 attention block,导致 recurrent 组与 full-attention 组持有不同边界,混合查找最终对齐到 0。前作 #54735 的堆叠后续。

## 优化点

机制上:为 `SingleTypeKVCacheManager` 增加 `has_positionally_stable_blocks` 属性,Mamba align 模式标记为位置不稳定,只有这类组跳过位置扫描、改用显式 boundary-state handoffs,Mamba `all` 模式与其他 append-only 组保持位置扫描;把外部接受的 token 范围按每组解析后的物理 block size 映射(细粒度块数向上取整,因外部命中按 hash block 对齐而非每组物理 block 对齐);eager store 的覆盖粒度与查找粒度对齐(不再把每组截断到 scheduler-block LCM);fine-grained 命中开启时在请求结束把 full-attention prompt 尾部也存入 CPU(SimpleCPU 本地位置查找,不占用共享 handoff 通道);lazy offload 模式禁用 partial hash hits 保持 scheduler-block 对齐;同一步内完成/被抢占请求的陈旧 handoff 被丢弃并按准入结果计数(新增 `BoundaryStoreStats` 可观测各丢弃原因);对同一请求同时贡献位置 attention 块与显式 Mamba 边界的 store 事件去重请求 ID。另修复 replicated 组被 DCP world size 过度缩放的问题(`dcp_world_size_for_kv_cache_spec`)。

## 关键改动

```text
vllm/v1/simple_kv_offload/manager.py: 边界 handoff 消费、按组块大小映射外部命中、BoundaryStoreStats 统计
vllm/v1/core/single_type_kv_cache_manager.py: MambaManager.has_positionally_stable_blocks(align 模式为 False)
vllm/v1/core/kv_cache_coordinator.py: group_block_sizes 暴露、allow_partial_hash_hits 开关
vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py: get_boundary_store_stats 诊断接口
tests/v1/simple_kv_offload/test_scheduler.py: 9 个混合场景调度器测试
```

核心判定:

```python
@property
def has_positionally_stable_blocks(self) -> bool:
    # Align-mode Mamba can null interior states and relocate speculative
    # blocks in place. Other modes retain positional identity.
    return self.mamba_cache_mode != "align"
```

## 性能声明

验证环境:source head `9cf76fe57`,eager SimpleCPU offload,`prefix_cache_retention_interval=0`。

- Kimi-K3 TP8/DCP8,确定性 dummy cold/hot/pressure/replay(32 GPU blocks):replay local hit 0/64543,replay external hit 64512/64543 = 99.95%,cold/replay 输出完全一致。
- Kimi-K3 真实权重 GSM8K 20-shot、1319 样本、并发 64、512 GPU blocks、负载驱动逐出:replay accuracy 0.9477(cold 0.9522),external hit 4048896/4693652 = 86.26%,local hit 0。
- Kimi-K3 AgentX/AIPerf fast,15 分钟、并发 16、512 GPU blocks(刻意压缩 GPU 池以逼出逐出):local hit 61.17%,external hit 39.08%,theoretical hit 95.38%,108 请求 0 错误。
- Qwen3.5-4B TP8/DCP2(8×B200,CUDA FLASH_ATTN,GDN-backed MambaSpec):dummy replay external hit 99.09%,输出一致;真实 BF16 20-shot GSM8K N20 并发 1(32 GPU blocks):cold/replay accuracy 均 0.80,external hit 93.44%,20/20 输出一致。

## 验证与回归风险

新增测试:`test_hybrid_store_uses_resolved_group_block_sizes`(DCP 复制组不过度缩放)、`test_dcp_mixed_cache_loads_fine_grained_external_hit`(非零本地前缀后接细粒度外部后缀)、finished eager store 的 FA partial tail 存取及去重、misaligned 本地前缀拒绝、lazy 模式禁用 partial hash hits、Mamba align 不做位置扫描/all 模式保留、离开请求的 handoff 丢弃、block_meta 平行索引。风险面:外部查找要求 num_computed_tokens 对 scheduler block 对齐(否则保留本地 hit 并告警跳过);传输仍为整块拷贝,不对齐时不启动跨块传输;影响 SimpleCPU offload connector 的 store/load 全路径,Mamba align 模式模型是主要受益者。

## 关联

堆叠于 #54735;同区域:索引中 KV offload/connector 相关卡(pr-24498 OffloadingConnector、pr-19330 KV load failure)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
