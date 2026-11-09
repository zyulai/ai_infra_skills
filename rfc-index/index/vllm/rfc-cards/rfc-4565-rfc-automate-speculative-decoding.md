# RFC #4565: [RFC]: Automate Speculative Decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#4565](https://github.com/vllm-project/vllm/issues/4565) |
| 作者 | LiuXiaoxuanPKU |
| 创建日期 | 2024-05-02 |
| 状态 | ⚪ closed (2025-03-20) |
| labels | RFC, stale |
| 热度 | 👍 32 · 💬 22 |
| 分类 | Speculative Decoding |

## 一句话结论
基于 goodput 动态决定每个请求的最优投机解码长度(0 到多 token),避免高负载或低接受率下延迟反升。

## 提案动机
投机解码是降延迟的关键特性,但在使用连续批处理的在线服务中并非总有效益:高请求率或低投机准确率时,延迟可能不降反升。提案希望智能决定每个请求的投机长度(从 0 即不投机到多 token),依据是反映全系统当前负载的 goodput 概念,从而让投机执行发挥最大效果;方法设计为通用的,兼容传统基于模型的投机到 prompt lookup、tree-style decoding 等无模型方法。

## 提出的方案
- **Milestone 1**:实现禁用投机解码的机制(proposed length = verified length = 0),让用户可手动停用;基于先前经验研究,先通过监控 `running_queue` size,超过阈值后对新请求暂停投机(Cody/@comaniac 协助实现)。
- **Milestone 2**:动态决定 proposed length——利用 batch size 等运行时信息,结合 profiled 参数(token acceptance rate、draft 与 target 模型的执行成本对比)实时调整,按当前系统状态优化性能。
- **Milestone 3**:消除对预 profiled 参数的依赖,直接从运行时收集 token acceptance rate 与 draft/target 执行时间(来自此前步骤),纳入 goodput 计算,使配置更动态、更响应。
- 已知限制:实现位于 scheduler 分配 slots 之后,可能造成内存低效(如 `num_lookahead_slots=5` 而 proposed length 仅 3 时,2 个 slot 闲置);当前仅支持 batch 级别的统一 proposed length,未来可考虑更细粒度。

## 优化点与收益
正文未给出量化数字。定性收益:在高请求率或低投机准确率场景避免投机解码带来的延迟恶化,按 goodput 自适应启停与调长,兼容多种投机解码风格;详细论文当时计划另行发布。

## 现状与落地
已关闭(2025-03-20),labels 含 stale。正文或结尾看不出「已实现/被替代/搁置」,落地情况需查关联 PR。

## 关联
正文未列出。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
