# RFC #22992: [RFC]: Refactor CI/CD

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22992](https://github.com/vllm-project/vllm/issues/22992) |
| 作者 | robertgshaw2-redhat |
| 创建日期 | 2025-08-15 |
| 状态 | ⚪ closed (关闭于 2026-01-23) |
| labels | RFC, good first issue, help wanted |
| 热度 | 👍 25 · 💬 17 |
| 分类 | Architecture / Engine |

## 一句话结论
全面重构 vLLM CI/CD:缩短单次提交 CI 时长、稳定测试、聚焦真正重要的模型与硬件。

## 提案动机
vLLM 的 CI/CD 多年累积后问题明显:CI 耗时很长(尤其每次提交周期);数值类失败无法在所有机器上复现;每次提交都跑非主流模型的测试;早期测试失败常导致清理不彻底、引发连锁失败难以定位;CI 未覆盖真正重要的模型(GPU 显存不够)与多数用户关心的硬件;CI 完全不覆盖性能。这些问题造成糟糕的开发者体验,形成"CI death spiral"(反复 force-merge 的循环),也让多个硬件后端难以获得可靠的阻塞信号。

## 提出的方案
- 移除 V0 测试,把缺失的覆盖迁移进 V1
- 将每次提交的 CI 端到端时长降到 <15 分钟
- 把所有依赖 golden strings 或数值的测试重构为稳定状态
- 模型测试重新聚焦到对 99% 用户重要的 top 10 模型
- 获取与用户实际部署模式匹配的硬件资源
- 作者强调这需要社区协同执行,并欢迎任何人参与,「We are starting next week」;反馈期 3 天

## 优化点与收益
正文给出的目标是「Reduce per-commit CI to <15 minutes end-to-end」(每次提交的 CI 端到端 <15 分钟);其余为定性收益:更稳定的测试信号、覆盖真正重要的模型与硬件、终结 force-merge 循环。未给出其他量化基准数字。

## 现状与落地
已关闭。正文为总体目标与行动倡议,未写明各项的实际完成情况,落地情况需查关联 PR。

## 关联
- #20218

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
