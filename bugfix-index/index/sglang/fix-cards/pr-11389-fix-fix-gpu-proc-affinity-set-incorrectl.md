# PR #11389: fix: fix gpu-proc affinity set incorrectly when pp_size > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11389](https://github.com/sgl-project/sglang/pull/11389) |
| 作者 | acelyc111 |
| 合并日期 | 2025-10-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`pp_size > 1` 时未把 pp 纳入 CPU affinity 计算,多进程被绑到重复的核集合;`set_gpu_proc_affinity` 引入 pp_size 修正每节点 GPU 数。

## 问题现象
`--pp-size > 1` 时 CPU affinity 设置错误。作者在 2 节点、pp=2 tp=8 启动时的日志显示,不同进程被绑定到重复的 CPU 核集合(cpu cores are bind duplicately),例如 TP0 与 TP4 都绑到 `[0..23, 96..119]`。打补丁后日志显示各进程绑定互不重叠的核集合(每进程 12+12 个核,如 TP0 → `[0..11, 96..107]`)。

## 根因
正文披露:启用 `SGLANG_SET_CPU_AFFINITY` 时未考虑 pp size,导致 pp>1 时 affinity 计算错误。修复前的计算为 `tp_size_per_node = tp_size // nnodes`(由 diff 推断:该式隐含假设 TP 组横跨全部节点,pp>1 时假设不成立),修复引入 `nnodes_per_tp_group = max(nnodes // pp_size, 1)`,以 `tp_size // nnodes_per_tp_group` 作为每节点 GPU 数。

## 修复方案
- `python/sglang/srt/utils/common.py`:`set_gpu_proc_affinity` 新增 `pp_size: int` 参数,计算改为

```python
nnodes_per_tp_group = max(nnodes // pp_size, 1)
tp_size_per_node = tp_size // nnodes_per_tp_group
```

- `python/sglang/srt/managers/scheduler.py` 与 `python/sglang/bench_one_batch.py`:两处调用改传 `server_args.pp_size`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:启用 `SGLANG_SET_CPU_AFFINITY` 时的进程 CPU 绑定计算(`run_scheduler_process` 与 `bench_one_batch.latency_test` 两个调用点);由 diff 推断,pp_size=1 时 `nnodes_per_tp_group` 仍等于 `nnodes`,单 PP 部署的计算结果与之前一致。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
