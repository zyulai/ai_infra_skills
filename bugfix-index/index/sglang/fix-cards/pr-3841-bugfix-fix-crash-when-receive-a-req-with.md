# PR #3841: [BugFix] Fix crash when receive a req with structed output in DP attention mode.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3841](https://github.com/sgl-project/sglang/pull/3841) |
| 作者 | hcyz33 |
| 合并日期 | 2025-02-25 |
| labels | (none) |
| 热度 | 👍 2 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论

DP attention 下收到结构化输出请求会崩溃;补上 idle batch 的 sampling_info_done 并改用 attn_tp_cpu_group 同步。

## 问题现象

开启 DP attention 后,收到带结构化输出(structured output)的请求会触发崩溃,对应 issue #3600。正文未描述具体报错日志与复现命令,未说明影响面大小。

## 根因

正文未披露根因,只在 Modifications 中列出两个修复点。结合 diff 可描述机制:

1. `process_batch_result` 的 idle batch 分支只调用了 `resolve_batch_result`,没有执行 `update_regex_vocab_mask()`、stream 同步和 `sampling_info_done.set()`,带 grammar/结构化输出请求的采样信息在 idle batch 场景下永远不置位(由 diff 推断)。
2. `move_ready_grammar_requests` 中 grammar 就绪数的跨 rank 同步用的是整个 `tp_cpu_group`;开启 DP attention 后参与同步的进程组与实际需要同步的 attn TP 组不一致,all_reduce 无法按预期完成(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/managers/scheduler.py` 两处:

1. `process_batch_result` 的 idle 分支补上采样信息同步:

```diff
             if self.enable_overlap:
                 self.tp_worker.resolve_batch_result(result.bid)
+                if batch.next_batch_sampling_info:
+                    batch.next_batch_sampling_info.update_regex_vocab_mask()
+                    self.current_stream.synchronize()
+                    batch.next_batch_sampling_info.sampling_info_done.set()
```

2. `move_ready_grammar_requests` 按 `enable_dp_attention` 分支:DP attention 时用 `attn_tp_cpu_group` 做 all_reduce(MAX)并按最大就绪数补全 grammar;否则保持原 `tp_cpu_group` 逻辑不变。另有一处 benchmark 脚本小改(`lines = list(lines)`)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:涉及 scheduler 的 idle batch 处理与 grammar 请求就绪同步两条路径;DP attention 开启时同步组从 tp 组换成 attn TP 组,非 DP attention 场景逻辑保持不变;idle 分支新增的 set 操作只在 `next_batch_sampling_info` 存在时执行。

## 关联

- https://github.com/sgl-project/sglang/issues/3600

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
