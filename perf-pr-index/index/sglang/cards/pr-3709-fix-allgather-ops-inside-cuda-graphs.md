# PR #3709: Fix allgather ops inside cuda graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @nvcastet |
| 合并 | 2025-02-25 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/3709 |

## 一句话结论
让 allgather 在 CUDA graph capture 内走 sglang 通信抽象(pynccl 优先),修复 capture 内外共用 NCCL communicator 的问题。

## 动机
SGLang 在 graph capture 时用 pynccl 或自定义 allreduce 而非 PyTorch ProcessGroup,但 allgather 直接调用了 `torch.distributed.all_gather_into_tensor`,导致同一个 NCCL communicator 会在 graph capture 内外同时被使用。allgather 用于 DP-attention 以及跨 TP 维 gather logits,因此该路径在 capture 下存在风险。该 PR 修复 issue #3424;正文无独立 Motivation 段落,以上为 TLDR 原文信息。

## 优化点
在 `parallel_state.py` 注册 `reg_all_gather_into_tensor` custom op,并给 `GroupCoordinator` 新增 `all_gather_into_tensor` 方法:pynccl 可用且未禁用时走 `pynccl_comm.all_gather`,否则回退 `torch.distributed.all_gather_into_tensor`。通过 custom op 封装使其可以在 graph capture 中安全使用,与 allreduce 已有的处理方式对齐。`GroupCoordinator.all_gather` 内部改为调用该抽象;`deepseek_v2.py` 的 DP-attention `all_gather` 也改走 `group.all_gather_into_tensor`,同时把 `self.tp_group` 从 `get_tp_group().device_group` 改为完整的 `get_tp_group()`。

## 关键改动

```text
python/sglang/srt/distributed/parallel_state.py: 注册 reg_all_gather_into_tensor custom op;GroupCoordinator 新增 all_gather_into_tensor(pynccl 优先)
python/sglang/srt/models/deepseek_v2.py: DP-attention 的 all_gather 改走 group.all_gather_into_tensor;tp_group 改为 GroupCoordinator
```

## 性能声明
PR 正文未提供 benchmark 数据(regex 亦未检出性能声明)。本 PR 属于正确性/兼容性修复(消除 capture 内外共用 communicator 的问题),性能收益未量化。

## 验证与回归风险
diff 中未包含新增测试,PR checklist 各项均未勾选。影响路径:DP-attention 的 allgather、跨 TP gather logits;行为变化在于 capture 内改用 pynccl allgather,若 pynccl 不可用则回退 torch.distributed,需要在真实 capture 场景下验证。

## 关联
修复 issue #3424;与 allreduce 在 CUDA graph 内的既有处理方式同一方向。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
