# PR #9285: Fixes a typo about 'max_decode_seq_len' which causes crashes with cuda graph.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9285](https://github.com/vllm-project/vllm/pull/9285) |
| 作者 | sighingnow |
| 合并日期 | 2024-11-08 |
| labels | ready |
| 热度 | 👍 0 · 💬 1 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
'max_decode_seq_len' 相关变量名拼写错误,导致超长序列仍被当作 CUDA graph 捕获处理而崩溃,修正拼写即可。

## 问题现象
超过 `max_seq_len_to_capture` 的序列仍被当作已由 CUDA graph 捕获来处理,进而导致崩溃。正文未提供具体报错栈与复现命令,仅描述该触发条件。

## 根因
正文披露:代码中存在一个关于 'max_decode_seq_len' 的拼写错误(typo),使得长度超过 `max_seq_len_to_capture` 的序列没有被正确排除出 CUDA graph 路径,仍按已捕获处理,最终崩溃。

## 修复方案
修正该 typo(正文标题即「fixes a typo about 'max_decode_seq_len'」)。
注:本 pack 的 diff 获取失败,无法给出具体改动的文件、函数与 diff 摘录;按正文描述,修复即纠正 max_decode_seq_len 相关的变量名拼写错误,使超长序列正确落入非 CUDA graph 执行分支。

## 影响与回归风险
- 回归面:仅纠正变量名拼写,涉及 CUDA graph 捕获长度判断;理论上只影响 typo 引入处附近的判断路径,回归风险低(diff 缺失,无法更精确评估)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
