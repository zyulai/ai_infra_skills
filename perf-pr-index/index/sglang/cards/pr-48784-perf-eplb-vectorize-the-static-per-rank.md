# PR #48784: [Perf] eplb: Vectorize the static per-rank dispatch map

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-11-06 |
| 分类 | Attention / Kernel(major) |
| Label | (none) |
| Diff | diffstat 缺失(素材未提供统计);diff 实际仅 1 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/48784 |

## 一句话结论

EPLB 静态 per-rank dispatch map 从逐元素 Python 循环改为张量化,DEP32 大拓扑 255→39 ms(6.5x)。

## 动机

`ep_dispatch_algorithm=static` 时,每次 EPLB rebalance 都要重建 per-rank dispatch map(`compute_logical_to_rank_dispatch_physical_map`),它对每个 (layer, logical expert, rank) 组合在 Python 里调一次 `_find_nearest_expert`。DeepSeek-V4.1-Flash 在 DEP32 下是 40 × 384 × 32 = 约 49 万次调用,每次 rebalance 在每个 rank 上耗 250-290 ms,期间整个 DP group 等待。

## 优化点

三步向量化,规则与原实现严格等价:(1) `_compact_candidates` 用 cumsum/scatter 把每行的有效物理 expert id 压到行首、保持原顺序,并记录有效数;(2) `_nearest_expert_dispatch` 用张量运算复刻 `_find_nearest_expert` 的选择规则——优先同 GPU 副本,否则当同节点副本能收窄候选集时选同节点副本(从宽到窄倒序扫 slot,使最早匹配的副本胜出),单副本的行直接广播;`_find_nearest_expert` 本身保留,继续服务启动期的 `_compute_logical_to_all_physical_map`;(3) 只有仍为 -1 的 (layer, expert, rank) 条目才回退到带 seeded `random.Random` 的 `_fair_choices`,且保持相同的行主序抽取顺序——因此输出与旧实现 bitwise 一致。scratch 张量建在输入 map 的 device 上;失去最后一个调用者的 `_logical_to_all_physical_raw` 被删除。

## 关键改动

```text
python/sglang/srt/eplb/expert_location.py: _compact_candidates + _nearest_expert_dispatch + 回退路径重组
```

```python
_, result = first_match(candidates // num_local_gpu_physical_experts, ranks)
if num_gpus_per_node is not None and num_local_node_physical_experts is not None:
    node_count, node_pick = first_match(
        candidates // num_local_node_physical_experts, ranks // num_gpus_per_node
    )
    narrows = (node_count > 0) & (node_count < num_valid[..., None])
    result = torch.where((result == -1) & narrows, node_pick, result)
```

## 性能声明

单次调用耗时:

| shape | before | after |
| --- | ---: | ---: |
| 40 层 × 384 logical / 416 physical,ep 32 | 255 ms | 39 ms |
| 3 层 × 128 logical / 160 physical | 5.6 ms | 9.1 ms |

大 shape(时间真正要紧的场景)6.5x;小 shape 因张量算子固定开销反而慢几 ms。E2E:DeepSeek-V4.1-Flash 分离式部署,GB300(DEP32 decode,8 节点),每 15000 迭代一次 EPLB,AgentX 3600 s、并发 3018,每 arm 单次同日运行:rebalance pause p50 0.734 → 0.467 s;每 GPU 总吞吐 458.7k → 466.6k(+1.7%);interactivity mean +2.4%、p90 +4.0%;TTFT p95 9.33 → 7.81 s。作者强调:pause 缩短 0.27 s 与单次调用节省一致、可归因于本改动;吞吐/interactivity/TTFT 的 delta 来自单次运行且大于 pause 本身能解释的量,不应读作本改动的效果;收益随 rebalance 频率与 EP 规模增长。

## 验证与回归风险

正确性以 bitwise 相等验证(placement 不变):96 个构造用例(12 布局 × 2 shape × 4 rank,scattered -1 padding,有无节点拓扑)+ 17,430 个随机布局 × rank(ep 1-16,1-8 节点,duplicate id,int32/int64)+ 12,910 个随机布局 × rank(ep 1-32,打乱候选顺序以覆盖 first-match 语义)。回归风险:小 shape 场景变慢几 ms;未解析条目的 `_fair_choices` 路径依赖行主序与随机种子一致性,若后续改动遍历顺序会产生与旧版本不同的 placement;`ep_size` 大时 `first_match` 的宽循环按候选宽度执行,行为随 topology 配置分支。

## 关联

#45282(non-blocking EPLB rebalance):把 rebalance 挪出 forward 线程但仍调用本函数,与本 PR 互补;其给同一函数新增 `device` 参数,与本 PR 有小冲突。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
