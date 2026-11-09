# PR #20697: Fix VRAM leak in overlap scheduling with structured output (#20640)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20697](https://github.com/sgl-project/sglang/pull/20697) |
| 作者 | Cishoon |
| 合并日期 | 2026-03-23 |
| labels | high priority, run-ci |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Overlap / Scheduler |
| bug 类型 | resource |

## 一句话结论
overlap 调度 + 结构化输出时,`delay_sample_func` 闭包持有 vocab_mask/next_token_logits 等大 GPU 张量直至下一迭代,VRAM 持续增长到 OOM;提前置 None 释放修复。

## 问题现象
开启 `extra_buffer`(overlap scheduling)并使用结构化 JSON 输出(grammar-based decoding)时,VRAM 稳定增长直至 OOM。正文明确泄漏只在两个条件同时满足时触发:一是 `extra_buffer` 开启 → overlap 调度 → 创建 `delay_sample_func` 闭包;二是请求结构化输出 → grammar 激活 → 每次迭代都分配新的 `vocab_mask`。单条件不满足时不泄漏。Fixes #20640。作者用复现脚本验证:修复后 `torch.cuda.memory_allocated()` 在迭代间保持稳定。

## 根因
正文披露:overlap 调度激活时,`tp_worker.py` 把 sampling 包装进 `delay_sample_func` 闭包,该闭包捕获 `forward_batch`(持有 `sampling_info.vocab_mask`)与 `logits_output`(持有 `next_token_logits`)。由于 `batch_result` 被 `result_queue` 和 `batch_record_buf` 保存到下一次调度迭代才释放,这些大 GPU 张量无法及时回收;每次迭代又分配新的 `vocab_mask`,VRAM 就持续增长直到 OOM。

## 修复方案
在张量不再需要的位置立刻释放引用,共三处:

- `python/sglang/srt/model_executor/model_runner.py` `_preprocess_logits`:`sampling_info.apply_logits_bias(...)` 之后立即 `sampling_info.vocab_mask = None`——mask 已施加到 logits,下游不再需要。
- `python/sglang/srt/managers/scheduler.py` `launch_batch_sample_if_needed`:闭包执行完后 `batch_result.delay_sample_func = None`,切断闭包对 `forward_batch`、`logits_output` 的引用。
- 同处:若 `batch_result.logits_output` 非空,置 `next_token_logits = None`——`copy_to_cpu()` 只拷贝 `next_token_ids` 与 logprob 字段,用不到原始 logits。

关键 diff 摘录(scheduler.py):

```diff
+        batch_result.delay_sample_func = None
+        if batch_result.logits_output is not None:
+            batch_result.logits_output.next_token_logits = None
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:改动仅两处赋 None 语句。`vocab_mask` 在 `apply_logits_bias` 之后、`next_token_logits` 在采样完成与 `copy_to_cpu` 之后才被清空,正常路径不读这些字段,行为不变;风险在于若后续代码新增对这些被清空字段的读取(如 logprob 相关路径依赖 `next_token_logits` 的扩展),会遇到 None,需注意空值保护。

## 关联
- Issue #20640(本 PR 修复的目标): https://github.com/sgl-project/sglang/issues/20640

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
