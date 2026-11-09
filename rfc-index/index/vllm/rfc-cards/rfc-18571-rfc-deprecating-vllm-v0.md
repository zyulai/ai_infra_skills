# RFC #18571: [RFC]: Deprecating vLLM V0

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18571](https://github.com/vllm-project/vllm/issues/18571) |
| 作者 | WoosukKwon |
| 创建日期 | 2025-05-22 |
| 状态 | ⚪ closed (关闭于 2025-12-30) |
| labels | RFC, stale |
| 热度 | 👍 64 · 💬 69 |
| 分类 | Other |

## 一句话结论
正式废弃并从代码库中移除 vLLM V0 引擎,全面转向 V1,以降低技术债与维护成本。

## 提案动机
V1 自 v0.8.0 发布起已成为默认引擎约三个月,用户采用广泛且反馈积极。V0 与 V1 共享大量代码(模型、配置、工具等),带来明显的复杂度与技术债:不熟悉 V0 的贡献者难以评估改动影响,新旧用户与贡献者(乃至 AI 助手)常混淆 V0/V1 模块。此外大量 CI 资源消耗在 V0 代码路径上,继续投入 V0 收益递减。

## 提出的方案
- 立即冻结 V0 功能开发,仅接受小的 bug 修复
- 2025-06-30 前完成关键特性向 V1 迁移:embedding 模型、Mamba 类模型、logits processors(新 API)、OpenTelemetry、Intel CPU 与 XPU 后端
- 部分特性暂停:encoder-decoder 模型(如 Whisper)、draft model 投机解码、Neuron 后端(计划在 V1 重引入)、HPU 后端(将以插件形式重引入)
- 部分特性永久移除:prompt adapter、V100 支持、per-request structured outputs 后端
- 版本时间线:v0.9 为最后支持 V0 的版本,v0.10 开始移除 V0 代码,v0.11 成为首个不含 V0 组件的版本

## 优化点与收益
正文给出预期:「Removing V0 is projected to reduce CI times and associated costs by at least half」(即 CI 时间与成本预计至少减半),条件是当前大量 CI 资源用于测试 V0 代码路径的背景;未给出具体基准模型/硬件/workload 的量化数字。

## 现状与落地
已关闭。正文为分阶段过渡计划(冻结 → 宣布废弃 → 移除代码与 CI 覆盖 → 更新文档),issue 已于 2025-12-30 关闭,但 pack 正文未写明计划实际执行结果,落地情况需查关联 PR。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
