# RFC #22932: [RFC]: Path to vLLM 1.0 Release

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22932](https://github.com/vllm-project/vllm/issues/22932) |
| 作者 | robertgshaw2-redhat |
| 创建日期 | 2025-08-14 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 21 · 💬 27 |
| 分类 | Other |

## 一句话结论
规划 vLLM 1.0 发布:借主版本升级清理外部接口、稳定并文档化可插拔接口、遵循语义化版本。

## 提案动机
vLLM 的 V1 架构已上线运行一段时间,V0→V1 迁移效果良好,当前正在弃用 V0 代码路径并清除技术债(见 V0 Deprecation Issue #18571)。作者提出应趁此规划"今年秋天某个时候"发布的 1.0 版本:V1 引擎更新的重心是重构与消除内部子系统的技术债,而 1.0 主版本升级则要顺势清理 vLLM 的外部接口并遵循语义化版本(semantic versioning)方案——随着 vLLM V1 中越来越多的接口变为可插拔,这一点愈发重要。

## 提出的方案
正在制定 1.0 发布工作计划,已确定的主要事项:
- 稳定并文档化现有可插拔接口:`Scheduler`、`KVConnector`、`ModelRunner`、`Model`;
- 澄清并文档化哪些子系统可插拔(public API)、哪些不可(private API);
- 稳定并文档化 `AsyncLLM` 与 `LLM` 接口,明确公开/私有边界;
- 重做并简化 vLLM 配置体系(EngineArgs、环境变量);
- 重做并简化日志(logging)体系;
- 简化 Prometheus 指标;
- 大修文档与 recipes;
- 大修性能跟踪系统;
- 官网(Website)。

## 优化点与收益
收益在工程与生态层面:公开/私有 API 边界清晰后,用户与下游集成者可依赖稳定的接口契约,语义化版本降低升级风险;配置、日志、指标与文档的简化降低使用与维护成本。正文未给出量化数字。

## 现状与落地
state=open:处于工作计划制定与意见征集阶段——反馈期设为 "End of August",清单各项目均未勾选,作者在正文末尾公开征集 "What else are we missing",并 CC 了多位核心维护者。

## 关联
- https://github.com/vllm-project/vllm/issues/18571 (V0 Deprecation Issue)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
