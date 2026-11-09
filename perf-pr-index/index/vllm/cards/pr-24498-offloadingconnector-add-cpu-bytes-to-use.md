# PR #24498: OffloadingConnector: Add cpu_bytes_to_use configuration

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @orozery |
| 合并 | 2026-01-12 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | documentation, ready, v1, kv-connector |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/24498 |

## 一句话结论

OffloadingConnector 的 CPU offload 容量配置从按 worker 的 num_cpu_blocks 改为实例级 cpu_bytes_to_use,块数由 KV cache 布局自动推导。

## 动机

PR 之前,`OffloadingConnector` 用 `num_cpu_blocks`(每 worker 的 CPU 块数)配置 offload 容量,用户需要换算 token 块数,难以按 vLLM 实例、跨 worker 直观地分配 CPU 内存;且 `num_cpu_blocks` 的正确计算依赖模型 KV cache 初始化之后的信息(代码中留有相关 NOTE),配置与实际布局脱节。

## 优化点

机制:配置语义从 `num_cpu_blocks`/`kv_bytes_per_rank`(按 rank 切分)改为实例级 `cpu_bytes_to_use`(总字节数,不做 per-rank 拆分),`VllmConfig._post_init_kv_transfer_config` 直接写入 `kv_offloading_size * (1 << 30)`。`CPUOffloadingSpec` 改为在构造时接收 `KVCacheConfig`,由 page size × kv_cache_tensors 数 × world_size × (offloaded_block_size / gpu_block_size) 推导每个 offloaded block 的字节数,再用 `cpu_bytes_to_use // kv_bytes_per_offloaded_block` 得出 `num_blocks`,传给 `CPUBackend`。为此 `OffloadingSpecFactory.create_spec` 与 `OffloadingConnector` 的构造链路都需要传入 `kv_cache_config`。LMCache 路径不受影响。

## 关键改动

```text
vllm/v1/kv_offload/cpu.py: CPUOffloadingSpec 接收 KVCacheConfig,由字节数推导 num_blocks
vllm/config/vllm.py: _post_init_kv_transfer_config 写入实例级 cpu_bytes_to_use
vllm/v1/kv_offload/spec.py / factory.py: OffloadingSpec 构造签名增加 kv_cache_config
docs/features/disagg_prefill.md: 示例改为 {"block_size": 64, "cpu_bytes_to_use": 1000000000}
```

## 性能声明

无性能声明;PR 正文与 diff 均未包含 benchmark 数据。此 PR 为 KV offloading 配置接口的现代化改造,不改变 offload 传输路径本身,PR 未声明任何性能数字或测试条件。

## 验证与回归风险

测试调整:`tests/v1/kv_connector/unit/test_config.py`(断言改为 `cpu_bytes_to_use`,且 TP2/DP2 时期望为全量 8 GiB 而非 per-rank 2 GiB)、`test_offloading_connector.py`(MockOffloadingSpec 签名)、`tests/v1/kv_offload/test_cpu_offloading.py`(配置键替换)。风险面:破坏性配置变更——`num_cpu_blocks`/`kv_bytes_per_rank` 不再被接受,现有用户脚本需改键名;`num_blocks` 推导依赖所有 kv_cache_group page size 一致(assert);非原生 spec(自定义 OffloadingSpec 子类)必须适配新构造签名。

## 关联

同区域:OffloadingConnector / kv_offload 体系的后续演进 PR;`--kv-offloading-size` 全局参数与 LMCacheConnectorV1 同属 KV Cache/Memory 分类。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
