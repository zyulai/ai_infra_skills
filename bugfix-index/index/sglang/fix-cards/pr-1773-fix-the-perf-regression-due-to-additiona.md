# PR #1773: Fix the perf regression due to additional_stop_token_ids

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1773](https://github.com/sgl-project/sglang/pull/1773) |
| 作者 | merrymercy |
| 合并日期 | 2024-10-23 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#1766 让每个请求把 additional_stop_token_ids 合并进 stop_token_ids 造成延迟回归,改为 stop_token_ids 置 None、判停时惰性读取。

## 问题现象
#1766 在延迟测试上引入性能回归,正文给出的复现命令是:

```
python3 -m unittest test_bench_serving.TestBenchServing.test_online_latency_default
```

正文未给出具体的延迟数字,只说明该测试出现 perf regression,本 PR 修复它。

## 根因
正文仅指出回归由 #1766 引入,未披露具体机制。从 diff 看,#1766 曾在 `sampling_params.normalize()` 中把 `tokenizer.additional_stop_token_ids` 合并进每个请求的 `stop_token_ids` 集合,且 `stop_token_ids` 默认初始化为空 `set()`;本 PR 改为:未传 stop_token_ids 时保持 `None`,`normalize()` 不再合并,判停时再惰性读取 tokenizer 上的集合(由 diff 推断:对每个请求做集合并拷贝是延迟劣化的来源)。

## 修复方案
- `python/sglang/srt/sampling/sampling_params.py`:`stop_token_ids` 无值时为 `None`;`normalize()` 删除合并 `tokenizer.additional_stop_token_ids` 的逻辑
- `python/sglang/srt/managers/schedule_batch.py` `check_finished()`:依次判断 `sampling_params.stop_token_ids`、`tokenizer.eos_token_id`、`tokenizer.additional_stop_token_ids`(均判空后取)
- `python/sglang/srt/hf_transformers_utils.py`:`handle_additional_stop_token_ids` 更名 `attach_additional_stop_token_ids`(get_tokenizer/get_processor 同步)
- `min_new_tokens.py` penalizer:`(stop_token_ids or set()) | (additional_stop_token_ids or set()) | {eos}`
- `sampler.py`:NaN 检测时先 `logger.warning` 再 `exit(1)`(顺序调整)

## 影响与回归风险
- 影响窗口:#1766 合并至本修复之间。
- 回归面:停止条件判定路径(check_finished、min_new_tokens penalizer、sampling_params 构造/normalize)——语义应等价但集合求值时机变化;函数改名影响直接引用 `handle_additional_stop_token_ids` 的外部代码;`stop_token_ids` 可能为 `None` 要求下游做判空(diff 中已同步处理)。

## 关联
- [#1766](https://github.com/sgl-project/sglang/pull/1766)(引入回归的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
