# PR #52921: [Bugfix] Align CPU offload pool size across PP/spec-decode workers

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52921](https://github.com/vllm-project/vllm/pull/52921) |
| 作者 | charxwu |
| 合并日期 | 2026-10-22 |
| labels | bug, ready, kv-connector, verified |
| 热度 | 👍 3 · 💬 56 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
不均 PP/spec-decode 下各 worker CPU offload 池块数不一致,调度器可能下发越界 block ID;worker 间 all-reduce MIN 对齐并上报告引擎统一使用。

## 问题现象
在 pipeline parallelism 不均或开启 speculative decoding 时,`SimpleCPUOffloadConnector` 的 CPU offload 块数在调度器与各 worker 之间不一致:更重的 PP rank 在相同字节预算下只能存更少的块,而调度器按较轻 rank 的容量估算池大小。调度器随后下发的 block ID 可能超出某个 worker 的实际分配,导致 offload 拷贝越界崩溃(copy operation crash)。

## 根因
正文披露:heavier PP ranks 在同一字节预算内能存储的块数更少,而调度器此前用一个较轻 rank 的容量来 sizing 自己的池;由此产生的 block ID 超出 worker 分配,拷贝时越界。worker 容量是从各自 live KV storage 推导的,天然按 rank 不同,却没有跨 rank 对齐机制。

## 修复方案
1. **Worker 侧 MIN 同步**:新增 `vllm/v1/simple_kv_offload/sizing.py` 的 `sync_num_offload_blocks_across_workers()`,在分配 offload 存储前对 world group 做 all-reduce `ReduceOp.MIN`,使每个 rank 分配相同的池大小;dist 未初始化或 world_size≤1 时为 no-op。`SimpleCPUOffloadWorker._resolve_num_cpu_blocks` 在无显式 override 时改走该同步。
2. **引擎侧查询**:`EngineCore._initialize_kv_caches` 仅当启用了 `SimpleCPUOffloadConnector`(含嵌在 `MultiConnector` 内,经新增 `_find_simple_cpu_offload_connector` 递归查找)且 prefix caching 开启时,经 RPC 查询 worker 对齐结果并断言有效("workers must report"),写入 `cache_config.simple_cpu_offload_num_blocks`(并排除出 config hash)。
3. **调度器侧直接使用**:`SimpleCPUOffloadScheduler` 新增 `aligned_num_cpu_blocks` 参数,无显式 `num_cpu_blocks` 时按对齐值使用;移除原来基于 config 的 fallback 与逐 worker 配置传递。显式 `num_cpu_blocks` override 仍受支持。
关键 diff 摘录(vllm/v1/executor/abstract.py):
```python
+        aligned = min(nums)
+        if len(set(nums)) > 1:
+            logger.warning(
+                "SimpleCPUOffload workers reported mismatched num_cpu_blocks %s; "
+                "using min=%d",
+                nums, aligned,
+            )
```
另附 cherry-pick 自 #62879 的 Nixl CI 测试脚本改动(共享 GPU 时先等 prefiller 就绪再启动 decoder,避免内存 profiling 与权重加载重叠)。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响启用 `SimpleCPUOffloadConnector` 且 PP 不均或 spec-decode 的部署。
- 回归面:`vllm/v1/simple_kv_offload/`(worker/manager/新 sizing.py)、`kv_transfer/.../simple_cpu_offload_connector.py`、`v1/engine/core.py`、`v1/executor/abstract.py`、`gpu_worker.py`、`config/cache.py`(新字段,参与 hash 排除)。验证:Kimi-K3 TP4×PP2 worker 50/46、scheduler 50 → 全部 46,两趟生成一致;TP8×PP2 双节点 MI355X 809/746 → 全部 746;CPU 测试 144 passed/18 skipped(GPU 相关 worker 同步测试需 CUDA/ROCm,AMD GPU 用例待 CI 验证;ROCm smoke 用 dummy weights)。

## 关联
- #62869(AMD fixture fix,CI 依赖)
- #62879(Nixl CI 测试脚本 cherry-pick 来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
