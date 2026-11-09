# RFC #23431: [RFC]: Continuous profiling and regression prevention for vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23431](https://github.com/vllm-project/vllm/issues/23431) |
| 作者 | linzebing |
| 创建日期 | 2025-08-22 |
| 状态 | ⚪ closed(关闭于 2026-01-30) |
| labels | RFC, stale |
| 热度 | 👍 11 · 💬 4 |
| 分类 | Other |

## 一句话结论
提议为 vLLM 建立持续性能剖析与回归防护体系,把性能优化成果长期"守住"。

## 提案动机
vLLM 在持续推进性能优化,但实际中 commit 可能悄悄造成引擎性能回退。作者观察到过去一个月引擎性能呈下滑趋势(以 vLLM Llama 8b shareGPT benchmark dashboard 的曲线为证)。为此需要:统一优化度量流程、长期保持优化成果并防止性能缓慢流失、降低度量与排查回退的运维开销(DevX)。

## 提出的方案
- 在 vLLM 内集成轻量级剖析插桩,配合周期性 e2e profiling 任务。
- 四个组件设计:
  - **vLLM Metric Collector**:以 hooks/timers 形式、由环境变量控制的可插拔插桩层,导出延迟分项分解(latency component breakdown),可扩展到 PD/RL/disagg 场景。
  - **性能度量工具**:提供 pre-commit 脚本,运行指定性能基准并输出详细延迟分解与吞吐对比差异。
  - **数据存储与可视化**:profiling trace 持续导出到 S3(6 个月 TTL);benchmark 结果落盘后上传 S3,由 AWS Lambda 解析写入托管 Clickhouse,用 Grafana 可视化各组件延迟随时间的趋势。
  - **回归检测告警与二分**:扩展现有"对任意 commit 触发 benchmark flow"的能力,按延迟或吞吐指标自动二分定位肇事 commit,并设置性能回退告警。
- 同时扩展现有 benchmark dashboard 纳入 SGLang 在部分 model/workload 组合上的数据,以便对照 vLLM vs. SGLang 的性能趋势。

## 优化点与收益
正文未给出量化数字。收益定位在工程流程:防止 commit 静默回退性能、缩短排查时间(自动二分)、降低度量优化的运维开销;现有基础设施(PyTorch 团队的 vLLM performance benchmark dashboard)已每 2 小时运行预定义基准,本 RFC 在其之上补充剖析数据与告警/二分能力。

## 现状与落地
已关闭(2026-01-30),labels 含 stale。正文中未说明实现进度,落地情况需查关联 PR。

## 关联
- [vLLM performance benchmark dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)
- https://github.com/Jialin/vllm/pull/1
- https://github.com/Jialin/vllm/pull/6
- [RELEASE.md manual validations](https://github.com/vllm-project/vllm/blob/main/RELEASE.md#manual-validations)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
