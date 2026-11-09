# RFC #5423: [RFC]: Improve guided decoding (logit_processor) APIs and performance.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#5423](https://github.com/vllm-project/vllm/issues/5423) |
| 作者 | rkooo567 |
| 创建日期 | 2024-06-11 |
| 状态 | ⚪ closed(关闭于 2025-04-17) |
| labels | RFC, stale, structured-output |
| 热度 | 👍 0 · 💬 16 |
| 分类 | Other |

## 一句话结论
提议重构 guided decoding/logit processor API:支持 async 与批处理、接入 SamplingParams,并让失败不再拖死 engine。

## 提案动机
当时 guided decoding 与 logit processor API 不完整,存在三大问题:guided decoding 参数不被 SamplingParams 支持;无法支持批式/异步 logit 处理;一旦失败 engine 会直接挂掉。部分问题可能已有对应 PR 在解决,本 RFC 汇总问题与方案。

## 提出的方案
- **API**:SamplingParams 支持问题已由 PR #4130 解决。
- **性能**:现状 logit processor 逐行(row by row)阻塞式应用;提议改用并行处理(如 Ray 或线程池)提升 logit 处理性能——Anyscale 内部已在用该机制,希望开源化,并让 API 支持 async 与 batching。
- **有状态 + prepare/apply 拆分**:logit processor 需要有状态(承载 Ray actor/线程池),PR #5329 应已足够;提出 `LogitPostProcessor` 接口:`initialize`(初始化状态)、`prepare`(异步准备 logit masks)、`apply`(把准备好的 mask 应用到 logits);`prepare`/`apply` 假定 1:1 调用。每个模型将增加 `prepare_logits` API,与 `compute_logits` 分离。
- **考虑上游化**基于 Ray 的批处理实现(配合 lmformatenforcer)。
- **失败处理**:有状态 logit processor 下请求可能失败(Ray actor 挂掉、schema 问题无法提前捕获);在 sequence.py 引入新状态 `FINISHED_INTERNAL_ERROR = enum.auto()`,任一 logit processor 失败即把相关 seq_group 标记为失败并中止该请求,engine 不再死亡。

## 优化点与收益
正文未给出量化数字。收益定位:logit 处理由逐行阻塞改为并行(Ray/线程池)批式+异步处理,提升 guided decoding 性能;同时改善鲁棒性(单请求失败不再导致 engine 死亡)。

## 现状与落地
已关闭(2025-04-17),labels 含 stale。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/pull/4130
- https://github.com/vllm-project/vllm/pull/5329
- lmformatenforcer(拟配合的 Ray 批处理实现)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
