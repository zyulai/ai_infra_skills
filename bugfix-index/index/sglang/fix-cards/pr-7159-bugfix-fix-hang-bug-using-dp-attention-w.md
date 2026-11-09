# PR #7159: [Bugfix]Fix hang bug using dp attention with HiRadixCache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7159](https://github.com/sgl-project/sglang/pull/7159) |
| 作者 | LLLL114 |
| 合并日期 | 2025-06-20 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
DP attention 下 HiRadixCache 误用 tp_cpu_group 导致 hang,按是否开 DP attention 选 attn_tp 组。

## 问题现象
正文未描述具体现象。标题与关联 issue 表明这是「使用 DP attention + HiRadixCache 时的 hang bug」,对应 issue [#7158](https://github.com/sgl-project/sglang/issues/7158)。触发条件、报错日志、影响面正文均未展开(Motivation 与 Modifications 小节为空)。

## 根因
正文未披露根因(正文仅写「Fix #7158」,Motivation/Modifications 留空)。

从 diff 看,改动位于 `scheduler.py` 的 `init_memory_pool_and_cache`:原代码构造 `HiRadixCache` 时把 `tp_cache_group` 一律传成 `self.tp_cpu_group`;修复后按 `server_args.enable_dp_attention` 选择 `self.attn_tp_cpu_group`(开 DP attention 时)或 `self.tp_cpu_group`。可推断机制是:开启 DP attention 时,HiRadixCache 用于跨进程 KV cache 协调的进程组应当是 attention TP 组,固定传入普通 TP 组导致同步对象不一致,进而挂起(由 diff 推断)。

## 修复方案
改动 `python/sglang/srt/managers/scheduler.py` 中 `init_memory_pool_and_cache` 对 `HiRadixCache` 的构造参数:

```python
tp_cache_group=(
    self.attn_tp_cpu_group
    if self.server_args.enable_dp_attention
    else self.tp_cpu_group
),
```

即开启 DP attention 时把 HiRadixCache 的 cache 组指向 `attn_tp_cpu_group`,否则维持 `tp_cpu_group`。其余参数(`page_size`、`hicache_ratio`、`hicache_size` 等)不变。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响同时启用 DP attention 与 HiRadixCache 的部署中 HiRadixCache 的跨进程通信组选择;不开 DP attention 的路径完全不变。若存在其他依赖 HiRadixCache 与 tp_cpu_group 绑定假设的逻辑,行为可能受影响,diff 范围内未见其他改动。

## 关联
- [#7158](https://github.com/sgl-project/sglang/issues/7158)(被修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
