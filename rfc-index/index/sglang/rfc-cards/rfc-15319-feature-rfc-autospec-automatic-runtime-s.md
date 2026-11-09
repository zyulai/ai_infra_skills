# RFC #15319: [Feature] RFC: AutoSpec, Automatic Runtime Speculative Inference Parameter Tuning

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15319](https://github.com/sgl-project/sglang/issues/15319) |
| 作者 | maodoudou168 |
| 创建日期 | 2025-12-17 |
| 状态 | ⚪ closed(关闭于 2026-03-28) |
| labels | (none) |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Speculative Decoding |

## 一句话结论
AutoSpec:用反馈控制器在运行时自动调 speculative_num_steps,免去手动调参并按负载自适应。

## 提案动机
用户须手动调 speculative_num_steps、speculative_topk、speculative_num_draft_tokens 三个参数。正文给出三个问题:逐模型/硬件/workload 试错开销大,且常得到次优配置;不同 draft 模型有效上限不同,静态参数无法适配;最优投机步数随 batch size 增大而下降(算力约束)。单一固定配置无法在变化的模型、硬件与 batch size 下都表现良好。

## 提出的方案
- 轻量反馈控制器实时调整 speculative_num_steps;为简单稳定,固定 speculative_topk=1、speculative_num_draft_tokens=speculative_num_steps+1(沿用观察到的最佳实践)。
- 每个 forward 解码步监控 acceptance rate:接受率高(可配置,如 ≥0.6)且未超当前 batch size 的硬件上限则增步;接受率低(如 <0.5)则减步;否则保持,形成收敛到当前 workload 近优步数的负反馈环,并以保守策略防振荡。
- 初始化阶段:系统启动时捕获一系列 cuda graphs(步数后续会动态变化),各 batch size 的投机参数用默认值或用户配置文件初始化。
- CLI:--auto-spec(默认 false)开启;--speculative-config-file 按模型/batch size 给出 steps 支持范围;--save-tune-results 保存各 batch size 下各 steps 的命中次数统计。
- 核心组件 AutoTunerEagle:集中管理参数注册表与调整逻辑,收集实时 acceptance rate 并执行调参决策。
- 分期:Phase 1 面向 EAGLE3 的动态参数调整与实时指标;Phase 2(可选)探索用 confidence/entropy 信号做 draft 模型提前终止。

## 优化点与收益
正文给出的数字:调优好的投机推理参数配置可提升吞吐 5%~50%(出自正文展示的"吞吐随 speculative_num_steps 与 batchsize 变化"实验图;正文未给出具体模型与硬件条件)。AutoSpec 的目标即自动逼近该调优收益,免去人工试错。

## 现状与落地
已关闭(2026-03-28)。正文为提案(Implementation Outline 两阶段,Related resources 为空),是否实现需查关联 PR。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
