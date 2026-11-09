# RFC #21796: [RFC]: Optimize embedding task

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#21796](https://github.com/vllm-project/vllm/issues/21796) |
| 作者 | maxdebayser |
| 创建日期 | 2025-07-29 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 10 · 💬 23 |
| 分类 | Other |

## 一句话结论
提议为 vLLM embedding 任务补齐基准测试、复现外部 4x 优化结果,并探索提升 GPU 利用率的方案。

## 提案动机
自 PR #21270 起 vLLM 完整支持 encoder embedding 模型(含 BERT 类模型及 gte-Qwen2-1.5B-instruct 这类由 decoder 改造的 encoder);encoder 双向注意力在 RAG 等信息检索场景优于 causal attention 的 decoder。但 Snowflake 的博客显示,他们对其后端(vLLM 位于 gRPC 前端之后)做的一系列优化将 seq len 512 的吞吐提升了 4x(基线为 V0 引擎跑 snowflake-arctic-embed-m-v1.5,BERT 模型,博客发表于 2025 年 5 月,当时 V1 尚未合入 main),其中一项优化虽是 Snowflake 特有基建,其余仍提示 vLLM 有改进空间。作者指出:vLLM 从未把 embedding 基准加入 `benchmarks` 脚本目录,甚至说不清 V0 当时表现如何,V1 是否更好、还有多少提升空间均是未知数。

## 提出的方案
RFC 旨在征集"如何做基准测试、尝试哪些优化"的意见,初始路线图:
1) 添加 embedding 基准测试脚本;
2) 尝试复现 Snowflake 公布的 vLLM V0 基线结果;
3) 对比 V1 与 V0 的性能;
4) 分析 V1 结果,排查 tokenization 及其他 CPU 任务造成的 GPU "bubbles"(空泡)。
另外,大多数 embedding 模型较小且不支持 chunked prefill 等优化,GPU 利用率可能欠佳;Snowflake 的解法是在单 GPU 上跑多个模型实例——可用数据并行探索,也可研究调度器(scheduler)与模型运行器(model runner)层面的优化来提升 GPU 利用率。

## 优化点与收益
潜在收益是 embedding 服务的吞吐与 GPU 利用率:Snowflake 博客显示其后端优化把 seq len 512 吞吐提升 4x(条件:模型 snowflake-arctic-embed-m-v1.5,BERT;基线为 2025 年 5 月的 vLLM V0 引擎)。vLLM 侧最终收益数字待基准建立后确定,正文未给出 vLLM 自身的量化数字。

## 现状与落地
state=open:处于意见征集与路线图提议阶段,反馈期为两周,已 CC 多位核心维护者;基准脚本与优化项均待实施。

## 关联
- https://github.com/vllm-project/vllm/pull/21270 (encoder embedding 模型完整支持)
- https://www.snowflake.com/en/engineering-blog/embedding-inference-arctic-16x-faster/ (Snowflake 博客)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
