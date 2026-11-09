# RFC #19407: [RFC]: Support automatic max context length via `--max-model-len auto`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#19407](https://github.com/vllm-project/vllm/issues/19407) |
| 作者 | yeqcharlotte |
| 创建日期 | 2025-06-10 |
| 状态 | ⚪ closed(关闭于 2026-01-07) |
| labels | RFC, stale |
| 热度 | 👍 10 · 💬 6 |
| 分类 | Feature / Ecosystem |

## 一句话结论
提议 `--max-model-len auto`,按 HBM 可用容量自动截断最大上下文长度,免去新模型首启必失败的困扰。

## 提案动机
vLLM 引擎要求显存至少能服务 1 个 `max-model-len` 请求;未显式设置时该值直接取自模型配置。随着 10M 乃至无限上下文模型日益普遍,这一默认行为越来越容易踩坑:正文举例,用 `vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct -tp 8` 启动 llama4 scout(10M 上下文),初始化约 10 分钟后报错——模型最大序列长度 10485760 需要 240.00 GiB KV cache,而可用 KV cache 仅 88.38 GiB,按可用显存估算的最大模型长度为 3861424。此前的 PR #16168 让用户调整 max-model-len 更方便,但尝试新模型时仍需经历一次失败;为多硬件、多并行配置维护该覆盖值也很麻烦,且用户常因此对 OOM 产生困惑。

## 提出的方案
- 支持 `--max-model-len auto`:自动把 max-model-len 截断到 HBM 容量可支持的最大上下文长度。
- 覆盖生效时向用户发出警告(warn users about overrides)。
- 实际改动需要对初始化代码做一些重构,确保更新正确传播到 `SchedulerConfig`、`CacheConfig` 等配置。
- 作者在正文中表示正在征求社区对该想法的反馈(Looking for feedback on the idea!)。

## 优化点与收益
属于易用性/健壮性改进而非性能优化:消除新模型(尤其超长上下文模型)首次启动必然失败、需手动重试一次的问题,并减少因该默认行为导致的用户 OOM 困惑。正文未给出量化数字(错误信息中的 240.00 GiB / 88.38 GiB / 3861424 为 llama4 scout + tp 8 下的报错示例,非性能收益)。

## 现状与落地
已关闭(2026-01-07),labels 含 stale;正文阶段为征求意见(Feedback Period 7/24),关闭时是否落地、是否被其他实现替代,从正文无法看出,落地情况需查关联 PR。

## 关联
- [PR #16168](https://github.com/vllm-project/vllm/pull/16168)(此前让 max-model-len 更易调整的工作,作者 @lengrongfu、@heheda12345)
- [kv_cache_utils.py 相关代码](https://github.com/vllm-project/vllm/blob/5f1ac1e1d14ce239ea6ea99e276ad90cdc342132/vllm/v1/core/kv_cache_utils.py#L562)(引擎至少需服务 1 个 max-model-len 请求的约束出处)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
