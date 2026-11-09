# PR #44784: [Bugfix] Prevent cuMemcpyBatchAsync segfault with MTP and KV offloading

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44784](https://github.com/vllm-project/vllm/pull/44784) |
| 作者 | JOSH1024 |
| 合并日期 | 2026-06-16 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 2 · 💬 59 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
KV offloading 误把 MTP draft 组纳入 load/store 调度导致越界块号,触发 cuMemcpyBatchAsync 段错误;修复为跳过 eagle 组并加 DMA 前越界检查。

## 问题现象
使用原生 KV offloading(`--kv-offloading-backend native`)并同时开启 MTP 投机解码时,在长上下文场景下 vLLM 直接崩溃,报 `cuMemcpyBatchAsync` segfault。崩溃是静默的 CUDA 段错误,缺少 Python 层报错,难以定位。

## 根因
正文披露:`OffloadingConnectorScheduler` 在 load 和 store 两条调度路径中都意外包含了 EAGLE/MTP draft-attention 组。这些 draft 组没有稳定的 block hash,且使用极小的 `gpu_block_size`,导致块数被 massively miscalculation(`num_gpu_blocks = cdiv(420K, small_size) = 105022`),进而产生越界的 GPU block index,在 `swap_blocks_batch` 中触发静默的 CUDA segfault。

## 修复方案
三处改动(正文 + diff):
1. 在 `GroupOffloadConfig`(vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)新增 `is_eagle_group` 标志,构造时按 `KVCacheGroupSpec.is_eagle_group`(或 EAGLE 投机配置)标记 draft 组,并打印检测日志。
2. 调度器在所有 load/store 调度循环中完全跳过 eagle 组:load 路径 `_lookup` 用 `eagle_verified` 集合保证 eagle 组只弹一次 volatile 尾块(滑窗组多查一块再 pop),store 路径 `_build_store_jobs` 对 eagle 组 `num_blocks = max(0, num_blocks - 1)` 排除尾块。
3. 正文所述的 defense-in-depth:在 `gpu_worker.py` 增加 DMA 前的边界检查,越界时抛出清晰的 Python `RuntimeError` 而非静默 CUDA segfault(该文件改动未出现在 pack 的 diff 摘录中)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:offloading connector 的 `_lookup` 收敛逻辑被改动(eagle 组引入 `query_max` 放大、`eagle_verified` 集合、非 eagle 组收紧时清空该集合),影响 KV offloading 的 load/store 命中计算;`is_eagle_group` 默认 `False`,不开启 EAGLE/MTP 时行为不变。

## 关联
Closes #44780

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
