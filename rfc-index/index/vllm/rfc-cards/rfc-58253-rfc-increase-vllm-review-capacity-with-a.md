# RFC #58253: [RFC]: Increase vLLM Review Capacity with a Reviewer Role

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58253](https://github.com/vllm-project/vllm/issues/58253) |
| 作者 | LucasWilkinson |
| 创建日期 | 2026-09-23 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 15 · 💬 3 |
| 分类 | Other |

## 一句话结论
提议设立 Reviewer 角色扩大评审容量:承担 triage 与初审、评审意见有约束力,但不能自行合入 main。

## 提案动机
vLLM 的贡献量超过评审与 triage 容量,PR 积压严重;大量低质量 AI 生成贡献("AI slop")加剧了识别有价值改动并把它们提升到项目标准的成本;今天这些责任压在还要兼顾实现、架构决策、发布与维护的 committer 身上。设立 Reviewer 也承认评审是宝贵贡献,并为走向 committership 提供清晰路径。

## 提出的方案
- Reviewer 可做:评审并给出被认可的 approve/request changes、触发 CI、把设计问题升级给对应 committer 与 SIG、triage PR/issue 并打标签、关闭重复 PR/issue、标记可进入 committer 评审的 PR;不可:自行合入 main(即使已批准且 CI 绿)
- 流程参照 Kubernetes reviewer/approver 模型:作者需先通过修改或回应解决 reviewer 反馈;outstanding concerns 须解决或由 committer 明确裁决后方可合入
- 任命:由 2 个不同组织的 committer 提名;经验法则:15 个 merged PR 且实质性 review 过 30 个 PR,或 review 过 45 个 PR;首次担任 reviewer 将成为新的 committer 的标准路径
- 公开 reviewer 名录与专长(docs.vllm.ai/en/latest/community/reviewers);在 #pr-reviews 公开协调;建立 Reviewer-Committer 私有 Slack 频道
- 实现选定 Option A:给 Reviewer Write 权限,受保护分支的更新限制在 Committer 团队与授权 bot,用原生 GitHub review 界面(Option B 为 Triage 权限 + bot 管理 review check,未选)
- Rollout:初始约 50 人小队,可能扩展到 2x active committers;评估首次实质评审时间、committer 终审等待时间、评审质量、CI 队列压力、积压规模与时长

## 优化点与收益
正文未给出量化数字。定性收益:贡献更早获得有效反馈、评审责任分散到更多可信贡献者、积压在不牺牲质量的前提下下降;先例为 Kubernetes、containerd、CPython triage team。

## 现状与落地
open。正文为完整提案(含已选定的实现选项与 rollout 计划),正在等待社区讨论与评审。

## 关联
- Kubernetes community-membership:https://github.com/kubernetes/community/blob/b8ce3416617273bfb8611b18b4cd8412893e44db/community-membership.md
- containerd GOVERNANCE:https://github.com/containerd/.project/blob/main/GOVERNANCE.md#reviewers
- CPython Triage Team:https://devguide.python.org/triage/triage-team/
- Reviewer 名录(计划):https://docs.vllm.ai/en/latest/community/reviewers

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
