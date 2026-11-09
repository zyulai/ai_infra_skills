# RFC #25144: [RFC]: vLLM's Path to Sustainable CI/CD

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25144](https://github.com/vllm-project/vllm/issues/25144) |
| 作者 | yeqcharlotte |
| 创建日期 | 2025-09-18 |
| 状态 | ⚪ closed (关闭于 2026-01-19) |
| labels | RFC, ci/build, stale |
| 热度 | 👍 13 · 💬 13 |
| 分类 | Other |

## 一句话结论
为 vLLM 提出可持续 CI/CD 的结构性改造:漏斗式测试分层、测试与代码同目录、flaky 隔离与自动化 triage。

## 提案动机
vLLM 的 CI 处于令人担忧的状态:慢、频繁失败、无法可靠检测关键模型的正确性与性能回退,导致开发者遇到测试失败时先怀疑 CI flakiness 而非代码。虽然 #22992 等工作正在推进(重构 golden tests、弃用 V0、改进 V1 覆盖、处理超时/flaky/硬件资源),作者认为还需要根本性的结构与流程变化,把 CI/CD 变成所有贡献者的共同责任,防止 CI 再次恶化。

## 提出的方案
- 五条原则:trunk 绿 >95% 时间;flaky 测试比没测试更糟;减少人工介入;优化 test<>fix 循环;测试被修改的内容
- 成功标准(buildkite):Reliability > 90%,Speed < 30min
- 漏斗式分层:precommit(<2min)/PR publish(5-10min)/PR ready(15-30min)/Nightly(2-6h)/Release(2-6h),前一层通过才安排更全面的测试;Nightly 覆盖 top 10 模型 E2E 性能与正确性(Hx00 与 GB200 为 P0)
- Issue 1 测试选择:tests/、microbench/ 下沉到各代码目录(按当前目录+2 层父目录选测试,默认约 5 分钟超时),新建 integration_tests/ 与 e2e_tests/,删除大 tests/
- Issue 2 生命周期:新测试先标记 unknown 不阻塞合入,持续通过(如 20 consecutive runs)转正;识别为 flaky 后在 CI 中跳过并向 owner 建单
- Issue 3 triage:已有测试失败自动 page CODEOWNERS,新测试失败 page 添加它的作者
- Issue 4 可复现:用 Justfile 存放测试执行命令作为唯一事实来源,作者可在本地复现

## 优化点与收益
照抄量化目标:Reliability > 90%、Speed < 30min、trunk 绿 >95% 时间;分层时间预算如 precommit <2 min、PR ready 15-30 min、Nightly 2-6 hours(每周 7 次)。

## 现状与落地
已关闭。正文末尾列出 Milestones(V0 完全弃用、搭建测试目录与 buildkite 初始结构、每 PR 增加 1 个模型的 E2E 正确性与性能测试),落地情况需查关联 PR。

## 关联
- #22992:https://github.com/vllm-project/vllm/issues/22992
- 治理文档 PR:https://github.com/vllm-project/vllm/pull/24801
- Buildkite test-state-and-quarantine 文档:https://buildkite.com/docs/test-engine/test-suites/test-state-and-quarantine
- 原始讨论文档:https://docs.google.com/document/d/1_wk0y4sPB1lPqDDHbuI1Dm-wUhztWjEstpoRt_aFluY/edit

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
