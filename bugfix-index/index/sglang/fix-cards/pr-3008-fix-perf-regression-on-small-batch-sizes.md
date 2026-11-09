# PR #3008: Fix perf regression on small batch sizes due to kv cache scale

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3008](https://github.com/sgl-project/sglang/pull/3008) |
| 作者 | merrymercy |
| 合并日期 | 2025-01-20 |
| labels | (none) |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Overlap / Scheduler |
| bug 类型 | regression |

## 一句话结论
#2786 引入的 kv cache scale 默认值 1.0 造成小 batch 性能回退;恢复 `None` 语义、仅在显式配置时原地除法。

## 问题现象
小 batch size 下出现性能回退。作者给出复现与量化数据:

```
python3 -m sglang.bench_offline_throughput --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-p 1 --random-input 32 --random-output 512 --random-range 1 --tp 8 --enable-torch-compile --cuda-graph-max-bs 1
```

修复前 438 token/s,修复后 449 token/s。

## 根因
正文仅说明该回退由 [#2786](https://github.com/sgl-project/sglang/pull/2786) 引入,未披露具体机制。(由 diff 推断)#2786 之后 `RadixAttention.k_scale/v_scale` 与 `set_kv_buffer` 的默认值成为 `1.0`,使 KV 写入路径(`cache_k.dtype != self.dtype` 分支)即使未配置 scale 也总是执行除法并生成新张量;修复恢复 `None` 默认,跳过无意义的缩放,并把除法改为原地操作。

## 修复方案
- `python/sglang/srt/layers/radix_attention.py`:`self.k_scale` / `self.v_scale` 由 `1.0` 改回 `None`;
- `python/sglang/srt/mem_cache/memory_pool.py`:`set_kv_buffer` 的 `k_scale` / `v_scale` 默认值改为 `Optional[float] = None`,仅当 scale 非 `None` 时执行原地 `cache_k.div_(k_scale)` / `cache_v.div_(v_scale)`,再统一 `.to(self.dtype)`。

```python
if cache_k.dtype != self.dtype:
    if k_scale is not None:
        cache_k.div_(k_scale)
    if v_scale is not None:
        cache_v.div_(v_scale)
    cache_k = cache_k.to(self.dtype)
    cache_v = cache_v.to(self.dtype)
```

## 影响与回归风险
- 影响窗口:正文说明回归由 #2786 引入,未给出具体版本范围。
- 回归面:KV 写入路径(`set_kv_buffer`)与 `RadixAttention` 默认 scale 字段;显式配置 k_scale/v_scale 的路径仍走非 `None` 分支执行除法,行为保留;变化的是未配置 scale 时的默认路径不再做除法。

## 关联
#2786

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
