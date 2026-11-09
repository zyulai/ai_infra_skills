# PR #14326: [Bugfix][Core] fix abort_seq_group and memory leak when n>1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14326](https://github.com/vllm-project/vllm/pull/14326) |
| 作者 | courage17340 |
| 合并日期 | 2025-03-06 |
| labels | ready |
| 热度 | 👍 4 · 💬 2 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
n>1 请求无法真正 abort 且 seq_id_to_seq_group 条目永不删除导致内存泄漏;abort 按组 ID 匹配并在完成时清理映射。

## 问题现象
存在 n>1 的请求时有两个 bug:
1. 无法正确 abort——日志打印 `Aborted request`,但请求实际仍在运行。复现:服务一个模型,curl 发送 n>1 且 stream=True 的请求,快速按 CTRL-C(建议设置较大 max_tokens 与 ignore_eos=True 以便复现)。修复前的日志显示 abort 后 metrics 中 `Running: 2 reqs` 持续存在、生成吞吐保持约 260 tokens/s;
2. `self.seq_id_to_seq_group` 中的元素永远不会被删除。

## 根因
正文披露的两点:n>1 的请求 abort 不生效(日志称已 abort 但实际仍在跑),以及 seq_id_to_seq_group 条目从不删除。匹配失败的具体机制(由 diff 推断,源自新增代码注释):n>1 时 `seq_group.request_id` 形如 `foo_parallel_sample_0`,而 abort 传入的 request_ids 只有 `foo`,二者不匹配,导致队列中的 seq group 逃过移除;同时 abort 与请求完成后都没有清理 `seq_id_to_seq_group`,条目随时间累积造成泄漏。

## 修复方案
修改 `vllm/core/scheduler.py` 与 `vllm/engine/llm_engine.py`:
1. `abort_seq_group` 新增可选参数 `seq_id_to_seq_group`;遍历队列时先经该映射把 seq_group 的 request_id 解析为 `real_request_id`(group_id)再与 request_ids 匹配;命中后不从 request_ids 移除 real_request_id(可能有其他 seq group 共享同一 ID);
2. abort 落地时删除 `seq_id_to_seq_group` 中对应条目;
3. `llm_engine.abort_request` 传入 `self.seq_id_to_seq_group`;
4. `llm_engine.step()` 中对 finished_requests_ids 清理 `seq_id_to_seq_group`,堵住内存泄漏。

关键 diff 摘录:
```python
if seq_group.request_id in seq_id_to_seq_group:
    real_request_id = seq_id_to_seq_group[
        seq_group.request_id].group_id
else:
    real_request_id = seq_group.request_id
if real_request_id in request_ids:
    aborted_groups.append(seq_group)
```

## 影响与回归风险
- 回归面:V0 调度器 `abort_seq_group` 的匹配逻辑与 `llm_engine` 的 abort/step 路径,均只在传入 seq_id_to_seq_group 时改变行为;n=1 请求 ID 解析结果与原先一致;新增的 step() 清理仅删除映射条目,不影响请求状态机。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
