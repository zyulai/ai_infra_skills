# RFC #62640: [RFC]: Improve MTP configuration warning

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62640](https://github.com/vllm-project/vllm/issues/62640) |
| 作者 | nfm-corti |
| 创建日期 | 2026-10-20 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
把 MTP "多次前向可能降低接受率"的激进警告改为更友好、可执行的提示(降为 INFO、改文案或抬高阈值)。

## 提案动机
当前 MTP 警告(`speculative.py:1375`)为:
```
WARNING ... Enabling num_speculative_tokens > 1
will run multiple times of forward on same MTP layer,which may result in lower acceptance rate
```
语气相当激进。但多数生产部署会用 `MTP >= 3`、更典型是 `MTP = 5`,vLLM 官方 recipes 也如此推荐;一条指出风险却不给任何具体建议的警告,是"不必要的 papercut"。

## 提出的方案
作者列出三个待讨论选项:
1. 把日志级别从 WARNING 降为 `INFO`;
2. 改进消息内容:指向 `https://recipes.vllm.ai/`,并提及 draft acceptance rate 指标、目标区间 60 - 80%;
3. 若保留 WARNING 级别,则把触发阈值提高到 `5`。

Feedback Period、CC List 等栏目为空,属于初步征求意见、尚无实现方案。

## 优化点与收益
正文未给出量化数字(收益为运维/可观测体验:减少生产部署的误报警噪音,并把用户导向 recipes 与 60 - 80% 的 draft acceptance rate 目标区间)。

## 现状与落地
仍在讨论/等待评审:open 状态,暂无评论(💬 0),正文以"想知道大家怎么看"收尾,尚未形成实现计划。

## 关联
- [vLLM recipes](https://recipes.vllm.ai/)(提案建议警告消息指向的站点)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
