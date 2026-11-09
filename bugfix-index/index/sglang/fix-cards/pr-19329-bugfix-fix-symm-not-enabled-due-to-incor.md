# PR #19329: Bugfix: fix symm not enabled due to incorrect registration of comm

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19329](https://github.com/sgl-project/sglang/pull/19329) |
| 作者 | wangfakang |
| 合并日期 | 2026-05-13 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 13 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`get_local_dp_buffer` 默认用 TP group 注册 symmetric memory,而实际 collective 用的是 attention CP/TP group,组不一致导致 symm 静默不生效;为 buffer 分配函数加 group 参数对齐注册组。

## 问题现象

symmetric memory(symm)优化未被启用——启用后不生效。正文未描述报错或更具体的运行时现象,直接归因为 comm 注册不正确。受影响的是 DP attention / CP 相关通信路径上由 `get_local_dp_buffer` / `get_global_dp_buffer` 分配、供 collective 使用的 buffer。

## 根因

正文披露的因果链:

1. `get_local_dp_buffer` 使用 `get_tp_group` 获得的 group 默认注册 symm;
2. 而 `attn_cp_all_gather_into_tensor` 等操作使用 `get_attention_cp_group` 获得的 group 执行 allgather;
3. 两者 group 不一致,启用 symm 时即失败(symmetric memory 要求注册组与执行 collective 的组一致,allocator 才能对相应组生效——正文以 `pynccl_allocator.py` 相关代码链接佐证)。

正文在 Motivation 中附了 5 处代码位置链接:`communicator_nsa_cp.py` L141-163、`dp_attention.py` L129-130 与 L572-573、`pynccl_allocator.py` L180-184 与 L52-58。

## 修复方案

给 `get_local_dp_buffer` 与 `get_global_dp_buffer` 增加 group 参数(类型为 `GroupCoordinator`;正文写作 GroupOrdinator),使 symm 注册的 comm group 与 collective 实际使用的 comm group 一致:

```diff
-    def get_global_dp_buffer(cls) -> torch.Tensor:
-        with use_symmetric_memory(get_tp_group(), disabled=not cls._dp_max_padding):
+    def get_global_dp_buffer(cls, group: GroupCoordinator) -> torch.Tensor:
+        with use_symmetric_memory(group, disabled=not cls._dp_max_padding):
```

各调用点传入与后续 collective 匹配的组:`communicator.py` 中 attn_tp allgather 处传 `get_attention_tp_group()`、dp gather 处传 `get_tp_group()`;`_scatter_hidden_states` 按 `get_tensor_model_parallel_world_size() == get_attention_dp_size()` 在 tp group 与 attention tp group 之间选择;`communicator_nsa_cp.py` 的 CP gather 处传 `get_attention_cp_group()`;MoE `token_dispatcher/standard.py` 的 flashinfer cutlass fp4 allgather combine 处传 `get_tp_group()`。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,`get_local_dp_buffer` / `get_global_dp_buffer` 签名变更,所有调用点(`communicator.py`、`communicator_nsa_cp.py`、`layers/moe/token_dispatcher/standard.py`)同步更新;影响启用 symmetric memory(`--enable-symm-mem` 类配置)且走 DP attention / NSA-CP / MoE combine 的部署——这些路径的 buffer 分配改在正确组上注册 symm。不改变 collective 语义本身,仅改变注册组;正文未附精度/性能数据。

## 关联

正文未列出 issue/PR 链接(Motivation 中引用了 5 处仓库代码行链接,见「根因」)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
