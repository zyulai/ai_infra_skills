# PR #3709: Fix allgather ops inside cuda graphs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3709](https://github.com/sgl-project/sglang/pull/3709) |
| 作者 | nvcastet |
| 合并日期 | 2025-02-25 |
| labels | high priority |
| 热度 | 👍 13 · 💬 26 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | general |

## 一句话结论
allgather 直调 torch.distributed 绕过 sglang 抽象,导致同一 NCCL communicator 在 CUDA graph capture 内外混用;改为统一走 GroupCoordinator 抽象。

## 问题现象
正文未描述具体报错或崩溃现象,只在 TLDR 中说明问题上下文:sglang 在 graph capture 时使用 pynccl 或自研 custom allreduce,而不是 pytorch ProcessGroup;但代码里的 allgather 直接调用 pytorch 的 allgather 实现,没有经过 sglang 的抽象去选择实现。Allgather 的使用场景是 DP-attention 和跨 TP 维 gather logits。

## 根因
正文披露:allgather 在 sglang 代码库里直接使用 pytorch 的 allgather,而不是通过 sglang 抽象决定用哪种实现,因此同一个 NCCL communicator 会在 graph capture 内部与外部被混用,这正是修复要避免的情形。

## 修复方案
改动两个文件:

1. `python/sglang/srt/distributed/parallel_state.py`:给 `GroupCoordinator` 新增 `_all_gather_into_tensor`(pynccl_comm 可用且未 disabled 时走 `pynccl_comm.all_gather`,否则回退 `torch.distributed.all_gather_into_tensor`),并注册 custom op `reg_all_gather_into_tensor`(含 fake impl),使该操作可被 graph capture 正确追踪;`all_gather()` 里原来的 `torch.distributed.all_gather_into_tensor(...)` 改为 `self.all_gather_into_tensor(...)`:

```python
def _all_gather_into_tensor(self, output, input):
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and not pynccl_comm.disabled:
        pynccl_comm.all_gather(output, input)
    else:
        torch.distributed.all_gather_into_tensor(output, input, group=self.device_group)
```

2. `python/sglang/srt/models/deepseek_v2.py`:`all_gather` 中同样改用 `group.all_gather_into_tensor(...)`;并把 `self.tp_group` 从 `get_tp_group().device_group` 改为 `get_tp_group()` 整个 coordinator 对象。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:DP-attention 与 logits 跨 TP gather 的 allgather 路径全部改走新抽象;不支持 custom op 的环境走 `supports_custom_op()` 回退分支;deepseek_v2 的 tp_group 语义从 raw device group 变为 GroupCoordinator,依赖该属性下游用法的一致性。

## 关联
- https://github.com/sgl-project/sglang/issues/3424

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
