# PR #10828: Revert "[fix][pd-disag]no need set next batch sampling info done in prefill"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10828](https://github.com/sgl-project/sglang/pull/10828) |
| 作者 | merrymercy |
| 合并日期 | 2025-09-24 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Sampler / Detokenization |
| bug 类型 | regression |

## 一句话结论
回滚 #10259:其跳过 prefill 阶段 regex vocab mask 更新的改动导致 CI 测试失败,恢复无条件执行。

## 问题现象
#10259([fix][pd-disag]no need set next batch sampling info done in prefill)合并后 CI 测试失败,失败任务链接为 https://github.com/sgl-project/sglang/actions/runs/17918784988/job/50995923340#step:5:1144。正文未描述失败测试的具体名称与报错细节,也未描述用户侧现象。

## 根因
正文未披露根因,仅认定 #10259 是 CI 失败原因并整体回滚。

从 diff 看,回滚的对象是 `scheduler.py` 的 `set_next_batch_sampling_info_done`:#10259 曾在 prefill 分离模式(`DisaggregationMode.PREFILL`)下跳过 `batch.next_batch_sampling_info.update_regex_vocab_mask()` 与 `self.current_stream.synchronize()`;回滚后恢复为只要存在 grammars 就无条件执行这两步。可推断机制是:prefill 模式下跳过 regex vocab mask 的更新会破坏 grammar 约束采样所依赖的掩码状态,从而在相关 CI 测试中失败(由 diff 推断)。

## 修复方案
改动 `python/sglang/srt/managers/scheduler.py` 的 `set_next_batch_sampling_info_done`,删除 prefill 模式的跳过分支,恢复无条件执行:

```python
if batch.next_batch_sampling_info.grammars is not None:
    batch.next_batch_sampling_info.update_regex_vocab_mask()
    self.current_stream.synchronize()
batch.next_batch_sampling_info.sampling_info_done.set()
```

即所有模式下 grammars 不为 None 都会调用 `update_regex_vocab_mask()` 并同步流,之后再 set `sampling_info_done`。

## 影响与回归风险
- 影响窗口:正文未说明(#10259 合入至本回滚合并之间)。
- 回归面:影响 PD 分离(prefill 端)下带 grammar 的请求路径——恢复 prefill 阶段的 regex vocab mask 更新与流同步;#10259 想省掉的 prefill 阶段同步开销会回来,prefill 端的 grammar 采样行为回到 #10259 之前。

## 关联
- [#10259](https://github.com/sgl-project/sglang/pull/10259)(被回滚的修复)
- [失败 CI 任务](https://github.com/sgl-project/sglang/actions/runs/17918784988/job/50995923340#step:5:1144)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
