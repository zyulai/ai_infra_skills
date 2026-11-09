# RFC #18037: [RFC]: Enabling Suffix Decoding, LSTM Speculator, Sequence Parallelism from Arctic Inference

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18037](https://github.com/vllm-project/vllm/issues/18037) |
| 作者 | sfc-gh-aqiao |
| 创建日期 | 2025-05-13 |
| 状态 | 🟢 open |
| labels | RFC, keep-open, speculative-decoding |
| 热度 | 👍 17 · 💬 15 |
| 分类 | Parallelism / Communication |

## 一句话结论
将 Snowflake Arctic Inference 的优化(Suffix Decoding、SwiftKV 等)以依赖方式引入 vLLM,并建立插件特性"毕业"进主库的通道。

## 提案动机
Snowflake AI Research 通过 Arctic Inference 插件为 vLLM 带来 Suffix Decoding、LSTM Speculation、Sequence Parallelism、SwiftKV 等优化,提升 TTFT、TPOT 与吞吐。但目前它是外部插件,用户需同时安装 vLLM 与 Arctic Inference 并附加配置;问题是如何让社区更方便地使用这些优化(如 Suffix Decoding 有普遍价值,而 SwiftKV 更偏模型特定),以及如何为特性从插件"毕业"到 vLLM 建立平滑路径。

## 提出的方案
- 列出三个选项:保持纯插件现状(最简单但用户需自行发现与安装);直接拷贝代码进 vLLM(与插件侧重复维护,且违背 vLLM 插件系统的设计初衷);将 Arctic Inference 作为 vLLM 依赖安装。
- **推荐方案 2.ii(作为依赖)**:Arctic Inference 划出 core 子模块(如 `arctic_inference.core`),保证接口稳定;次级依赖极少(当前为零,torch 等主要依赖均继承自 vLLM)。
- Suffix Decoding 的集成代码直接贡献进 vLLM,用户原生使用、零额外步骤;其余特性可通过启用 Arctic Inference 插件 opt-in 使用(默认关闭)。
- 该通道建立后,未来新特性只需在 vLLM 侧编写集成代码即可平滑"毕业"为原生功能。

## 优化点与收益
正文照抄的数字(Snowflake 工程博客口径,未给出模型/硬件细节):
- **4x faster generation** with Suffix Decoding for SWEBench
- **2.4x faster generation** with LSTM Speculator for ShareGPT
- **2.8x faster coding** with LSTM Speculator for HumanEval
- **2x higher throughput** with SwiftKV
- **1.4x throughput than TP=8, but same TTFT as TP=8** with Arctic Ulysses
这些优化面向 Snowflake 生产工作负载设计,并将持续扩展与维护。

## 现状与落地
仍在讨论/等待评审:正文给出推荐方案(option 2.ii)并 CC 了 @simon-mo、@WoosukKwon、@LiuXiaoxuanPKU 等维护者,等待社区反馈;具体落地需查后续关联 PR。

## 关联
- [Arctic Inference 仓库](https://github.com/snowflakedb/arcticinference)
- [Suffix/LSTM Speculator 博客](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- [SwiftKV 博客](https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/)
- [Arctic Ulysses 博客](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)
- [vLLM 插件系统文档](https://docs.vllm.ai/en/latest/design/plugin_system.html)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
