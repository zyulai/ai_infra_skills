# PR #44490: [Bugfix][Core] Fix host memory leak from undrained new_block_ids

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44490](https://github.com/vllm-project/vllm/pull/44490) |
| 作者 | Sunt-ing |
| 合并日期 | 2026-07-07 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 10 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
非 Mamba 模型 new_block_ids 只记不排致 host 内存线性泄漏;每步无条件 drain 修复。

## 问题现象
持续 `max_tokens=1` 负载下 host RSS 线性增长、无平台期:报告者 12 小时运行 Gemma-3-1b + prefix caching,RSS 约 6.0→6.57 GiB。受影响面:任何带 full-attention 或 MLA KV-cache 层且**无 Mamba 层**的 V1 模型——即几乎所有标准 attention 模型(Llama、Qwen、GPT/OPT、Gemma-2/3、DeepSeek-MLA);Mamba / hybrid-Mamba 模型不受影响。`EngineCore` 启动时的 `gc.freeze()` 使该 list 躲过 `gc.get_objects()`/`tracemalloc`,所以现象看起来像 allocator fragmentation。

## 根因
正文披露:#35219 引入的 `SingleTypeKVCacheManager.new_block_ids` 在每次 full-attention/MLA block 分配时追加,但只在 `needs_kv_cache_zeroing`(等于 `has_mamba_layers`)时才 drain:记录按 spec 类型(`FullAttentionSpec`/`TQFullAttentionSpec`/`MLAAttentionSpec`)门控,drain 却按「有无 Mamba 层」门控——非 Mamba 模型每次分配都记录但从不排出,list 无界增长(每请求每个分配 block 一个 int)。#35219 自述「非 hybrid(纯 attention)模型完全不受影响」,本修复使其成真。

## 修复方案
consumer 侧单文件修复,scheduler 每步无条件 drain:
```python
new_attn_block_ids = self.kv_cache_manager.take_new_block_ids()
new_block_ids_to_zero = (
    (new_attn_block_ids or None) if self.needs_kv_cache_zeroing else None)
```
`single_type_kv_cache_manager.py` 新增 `needs_kv_cache_zeroing` 参数,记录门控改为 `needs_kv_cache_zeroing 且 spec 类型匹配` 的 `_record_new_block_ids`;`kv_cache_coordinator.py` 构造 manager 时传入该参数。drain 出的 id 只被 zeroing 路径消费,无条件 drain 对所有模型安全;Mamba 模型原本就在该分支 drain,行为逐字节不变。作者特意不用新构造标志门控 recording:正确性特性(KV zeroing)的默认 False 标志会被忘传的构造路径静默禁用。验证:opt-125m 300k 请求 EngineCore VmRSS +167.4→+4.2 MiB(list 19.85M→0);gemma-3-1b 真实 `vllm serve` 400k 请求容器 RSS +163.8→+1.5 MiB;Qwen3.5-4B(hybrid Mamba)本就不漏,修复为 no-op;每项 before-fix 运行 `gc.collect()` 释放 0 MB,证实是使用中泄漏而非碎片。

## 影响与回归风险
- 回归面:`vllm/v1/core/sched/scheduler.py` 的 schedule()、`single_type_kv_cache_manager.py` 构造与两处记录点(allocate_external_computed_blocks / allocate_new_blocks)、`kv_cache_coordinator.py` 传参;对 `needs_kv_cache_zeroing=True`(Mamba)模型为 no-op。

## 关联
Fixes #44175;引用 #35219

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
