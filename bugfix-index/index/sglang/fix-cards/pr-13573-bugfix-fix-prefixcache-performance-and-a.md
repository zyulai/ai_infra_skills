# PR #13573: [BugFix] fix prefixcache performance and accuracy on ascend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13573](https://github.com/sgl-project/sglang/pull/13573) |
| 作者 | khalil2ji3mp6 |
| 合并日期 | 2025-12-07 |
| labels | deepseek, hicache, npu, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

Ascend 上 MLA 模型 prefix cache 存在性能与精度问题:预搬 device_indices、按设备选 pinned 内存分配方式,并接入 npu_ring_mla 修复精度。

## 问题现象

Ascend NPU 上 MLA 模型(DeepSeek)启用 prefix cache / 层级缓存(hicache)时存在性能与精度两类问题。性能基准:DeepSeek v3,3.5K GSM8K 数据集 50% 复用率(一半请求重复),对比禁用 radix cache 与开启 L1+L2 缓存,最终 TTFT 降低 39.5%。精度问题经新测试用例 `test_ascend_hicache_mla.py` 验证,结果与关闭该功能时一致。正文未给出修复前具体的错误输出。

## 根因

正文对根因着墨有限,仅按问题分类给出对应修法。性能方面(由 diff 推断):`cache_controller.py` 的 kernel_ascend 分支此前直接返回 NPU 上的 `device_indices`,未先搬到 CPU,造成传输路径上的额外开销;`memory_pool_host.py` 原先一刀切 `SUPPORT_PIN_MEMORY = not _is_npu` 并用 `cudaHostRegister` 锁页,NPU 上不适用。精度方面:正文说明通过集成 `npu_ring_mla` 算子对 attention 模块做分段计算来修复——暗示原有 ascend attention 路径在带 prefix(extend_prefix_lens > 0)时计算不正确,正文未进一步披露具体机制。

## 修复方案

性能两处:

1. `cache_controller.py`:`io_backend == "kernel_ascend"` 分支 `move_indices` 返回值改为 `device_indices.cpu()`,提前把设备索引搬到 CPU;
2. `memory_pool_host.py`:移除全局 `SUPPORT_PIN_MEMORY`,新增 `NEED_HOST_REGISTER` 语义的 `ALLOC_MEMORY_FUNCS`——默认用 `alloc_with_host_register`(`cudaHostRegister` 锁页),`npu` 设备改走 `alloc_with_pin_memory`(直接 `pin_memory=True` 分配);`HostKVCache.init_kv_buffer` 等 3 处分配点统一改经 alloc 函数。

精度一处:`ascend_backend.py` 在 `use_mla` 且 extend 带 prefix 时新增分支,用 `torch_npu.atb.npu_ring_mla` 两段式计算(先算 extend 部分,再加载历史 KV cache 经 `kv_b_proj` 变换后与 prefix 部分做 ring 计算),配套 `ForwardMetadata` 增加 `prefix_lens` 与 `flatten_prefix_block_tables`,并新增测试 `test/srt/ascend/test_ascend_hicache_mla.py` 纳入 `per-commit-4-npu-a2` 套件。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:host 内存分配路径对所有后端的 host KV cache 生效(非 NPU 设备仍走 `cudaHostRegister` 路径,行为不变;NPU 从「不锁页」变为 `pin_memory=True` 分配);ascend attention 的 extend-with-prefix 走全新 `npu_ring_mla` 路径;kernel_ascend 的 cache controller 索引搬运语义变化(返回 CPU 张量)。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
