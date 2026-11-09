# PR #15805: [HiCache] Fix deadlock when creating new group

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15805](https://github.com/sgl-project/sglang/pull/15805) |
| 作者 | XucSh |
| 合并日期 | 2025-12-29 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 16 |
| 分类 | KV Cache / Memory |
| bug 类型 | hang |

## 一句话结论
多节点 HiCache 初始化时各 rank 以不同 ranks 参数调用 new_group 这个集合操作,匹配失败死锁;改为全局收集配置后按统一顺序建组。

## 问题现象
多节点分布式环境(例如 2 节点、每节点 8 GPU)下,用 `--enable-hierarchical-cache --hicache-storage-backend` 启动服务时,初始化过程在 `torch.distributed.new_group` 处挂起(hang),这是 HiCacheController 初始化阶段的关键死锁。正文以 PP2 TP8 为例描述触发路径:Node 0(Ranks 0-7)与 Node 1(Ranks 8-15)各自创建本节点的 prefetch 通信组。

## 根因
正文明确披露:`torch.distributed.new_group` 是集合(collective)操作,要求默认 world group 中的所有 rank 以相同顺序、完全相同的参数(特别是 ranks 列表)调用。此前的 HiCacheController 实现中:

- Node 0 (Ranks 0-7) 调用 `new_group(ranks=[0...7])`
- Node 1 (Ranks 8-15) 调用 `new_group(ranks=[8...15])`

各 rank 参数不同,内部 distributed store 无法匹配创建请求,造成死锁:Node 0 等待 Node 1 确认创建 Group A,而 Node 1 正在尝试创建 Group B。

## 修复方案
- `python/sglang/srt/distributed/parallel_state.py`:新增 `create_custom_parallel_group(group_ranks, backend="gloo")`:先 `all_gather_object` 把每个 rank 想加入的 ranks 配置收集到全局,按签名去重、按首 rank 排序得到全局唯一的建组列表,然后所有 rank 按同一顺序对每个 unique group 调用 `torch.distributed.new_group`;若自己属于该组则保留返回值,否则返回 None。这样集合调用在所有 rank 上参数与顺序一致。
- `python/sglang/srt/managers/cache_controller.py`:`HiCacheController.__init__` 中 TP world size > 1 时,创建 prefetch 同步组改用 `create_custom_parallel_group(group_ranks=group_ranks, backend="gloo")`,替换原来的 `torch.distributed.new_group(group_ranks, backend="gloo")`。

```python
    torch.distributed.all_gather_object(gathered_configs, local_config)
    ...
    unique_groups.sort(key=lambda x: x[0])
    for g_ranks in unique_groups:
        group = torch.distributed.new_group(ranks=g_ranks, backend=backend)
        if set(g_ranks) == set(local_config):
            my_new_group = group
```

## 影响与回归风险
- 回归面:影响 HiCache(`--enable-hierarchical-cache`)多节点场景下 prefetch TP 组的创建路径(cache_controller.py)与 parallel_state.py 新增的通用建组助手;所有 rank 现在会为每个 unique group 各执行一次 `new_group` 集合调用(非本组成员也参与调用),组数量随拓扑变化,后端仍为 gloo。单节点(各 rank 配置一致)行为等价于原来只建一组。

## 关联
正文未列出(正文仅 @ 了报告者 @whybeyoung 与 @stmatengss)
