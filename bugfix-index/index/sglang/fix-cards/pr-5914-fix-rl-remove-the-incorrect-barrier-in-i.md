# PR #5914: [fix][RL] Remove the incorrect barrier in init_weights_update_group

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5914](https://github.com/sgl-project/sglang/pull/5914) |
| 作者 | zhuzilin |
| 合并日期 | 2025-05-15 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
RL 训练经自定义资源管理(如 ray)时 engine 拿不到 device rank,`init_weights_update_group` 里的 barrier 调用非法,直接删除。

## 问题现象
正文 Motivation/Modifications 均为空模板,未描述具体现象,只在 Modifications 中说明:该 PR 通过移除一个不正确的 barrier 修复 `init_weights_update_group`;触发条件是「用自定义资源管理(如 ray)做 RL 训练时,每个 engine 无法访问 device `rank`」。除此之外正文未给出报错信息或复现步骤。

## 根因
正文披露:`init_weights_update_group` 中的 `dist.barrier(group=self._model_update_group, device_ids=[rank])` 依赖本地 `rank` 作为 `device_ids` 参数;在 RL 训练使用自定义资源管理(例如 ray)时,每个 engine 并不持有 device `rank`,该 barrier 调用因此不成立(正文原话:"each engine will not have access to a device `rank`")。正文未进一步披露该调用在无对应 device 时会抛出何种异常,细节需以实际代码为准。

## 修复方案
- `python/sglang/srt/model_executor/model_runner.py` 的 `init_weights_update_group`:在成功初始化 custom process group 后,删除 `dist.barrier(group=self._model_update_group, device_ids=[rank])` 这一行,直接返回 `True, "Succeeded to initialize custom process group."`:
```diff
                 rank=rank,
                 group_name=group_name,
             )
-            dist.barrier(group=self._model_update_group, device_ids=[rank])
             return True, "Succeeded to initialize custom process group."
```
- `test/srt/test_update_weights_from_distributed.py`:同步删除 `init_process_hf` 中对应的 `dist.barrier(group=group, device_ids=[rank])`;并顺手修正 `init_process_sgl` 中 URL 拼接错误(`":".join(host, ...)` 改为 `":".join([host, ...])`,端口解析改用 `rpartition(":")`)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:按 diff,权重更新分组初始化(`init_weights_update_group`)不再执行组内 barrier 同步,依赖该 barrier 做 trainer 与推理 engine 间同步的调用方需自行保证时序;分布式权重更新测试路径同步变更。删除 barrier 属纯删减,不影响其他代码路径。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
