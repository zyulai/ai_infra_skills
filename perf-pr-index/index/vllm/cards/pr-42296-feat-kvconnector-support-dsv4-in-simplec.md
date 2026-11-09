# PR #42296: [Feat][KVConnector] Support DSV4 in SimpleCPUOffloadBackend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ivanium |
| 合并 | 2026-05-25 |
| 分类 | KV Cache / Memory |
| Label | ready, v1, kv-connector |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 内容本身可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/42296 |

## 一句话结论

SimpleCPUOffloadBackend 支持 DSV4:修正 block size 解析并跳过 SWA 块,多轮 workload 吞吐 +81.7%。

## 动机

SimpleCPUOffloadBackend(由 PR #42258 引入)此前假设 GPU 与 CPU 使用同一 block size,直接取 `cache_config.block_size` 做 hash 与换算。对 DSV4 这类模型,scheduler block size、hash block size 与 FA group 自身的 block size 并不相同,且 SWA 层部分块根本不会被 hash。这导致 eager 模式下前缀缓存命中与 offload 存储边界计算错误;lazy 模式作者称"天然正确",不受影响。

## 优化点

核心是把 CPU offload 的 hash/换算逻辑对齐到 vLLM v1 的多 block size 体系。其一,scheduler 角色初始化时用 `resolve_kv_cache_block_sizes` 解析出 scheduler_block_size(各 group 的 LCM)与 hash_block_size,替换原来对 `cache_config.block_size` 的直接假设,并断言两者整除关系。其二,单独记录 FA group 自己的 `fa_block_size`,命中换算 `num_computed_tokens` 时按 FA 的块大小计,而非假设与 scheduler 一致。其三,eager store 扫描前先把 confirmed tokens 对齐到 scheduler_block_size(`aligned_tokens`),再按各 group 的 block size 换算 ready_blocks,避免跨 group 错位。其四,扫描中遇到被 mask 掉、没有 hash 的 SWA 块时由 `break` 改为 `continue`,这类块不可能命中前缀缓存,跳过后继续扫描后续块,不再截断整个 store 扫描。

## 关键改动

```text
vllm/v1/simple_kv_offload/manager.py: 分离 scheduler_block_size/hash_block_size/fa_block_size,命中换算与 eager store 对齐、跳过无 hash 的 SWA 块
vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py: scheduler 侧解析并传入两个 block size
tests/v1/simple_kv_offload/test_scheduler.py: make_scheduler 增加对应参数
```

关键片段(manager.py,跳过不可能命中的 SWA 块):

```python
bhash_with_group = gpu_block.block_hash
if bhash_with_group is None:
    # Masked-out SWA position the coordinator chose not to
    # hash; it can never serve a prefix-cache hit, so skip.
    advanced_per_group[g] += 1
    continue
```

## 性能声明

DSV4-flash + 合成多轮(multi-turn)synthetic workload,baseline vs 本 PR("simple"):Output throughput 124.2 → 225.7 tok/s(+81.7%);Request throughput 0.621 → 1.129 req/s(+81.7%);Mean TTFT 8825.07 → 6452.24 ms(-26.9%);Median TTFT -47.1%;P99 TTFT 25599.02 → 26259.39 ms(+2.6%,轻微回退);Mean TPOT 83.03 → 35.61 ms(-57.1%);Mean ITL 84.10 → 39.51 ms(-53.0%);Mean E2EL 25348.12 → 13538.82 ms(-46.6%)。模型规模、硬件、精度等测试条件 PR 未注明。

## 验证与回归风险

测试:`tests/v1/simple_kv_offload/test_scheduler.py`、`tests/v1/simple_kv_offload/test_integration.py`(pytest -v),另跑 DSV4-flash 多轮 workload 验证 external cache hit。风险集中在 simple_kv_offload 路径:block size 语义变化影响所有使用该 connector 的模型配置;SWA 模型的 store 扫描行为改变(跳过无 hash 块);修复主要作用于 eager 模式;新增的 `block_size % hash_block_size == 0`、`block_size % fa_block_size == 0` 断言对不满足整除关系的配置会直接启动失败。

## 关联

依赖前序 PR #42258(需先合并);同属 kv-connector / CPU KV offload 区域,无其他明确关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
