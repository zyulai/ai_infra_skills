# RFC #8779: vLLM's V1 Engine Architecture

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#8779](https://github.com/vllm-project/vllm/issues/8779) |
| 作者 | simon-mo |
| 创建日期 | 2024-09-24 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 90 · 💬 14 |
| 分类 | Architecture / Engine |

## 一句话结论
提出 vLLM V1 引擎重构方向:以有状态 worker + 异步调度兼顾简洁与性能,降低 CPU 开销与 GPU 空闲时间。

## 提案动机
第一代引擎简洁但性能不足,multi-step 引擎高性能但复杂,新架构希望兼得两者;现有代码库因功能相互纠缠,添加生产或实验特性都变得困难;还需清理技术债,移除牺牲可读性的优化、重做临时性实现。范围限定在 scheduler、memory manager、分布式架构,不触及 API、模型与内核。

## 提出的方案
- **Driver 进程 + SPMD worker**:TP=n & PP=m 时共 n*m+1 个进程(即使单 GPU 也有 2 个进程);worker 有状态、维护大部分请求状态,driver 只发送"diffs"(新请求:输入 token IDs/block tables/sampling params;进行中请求:调度 ID 与新块 ID);清理 `SeqGroupMetadata` 等数据结构。
- **异步单步调度**取代 multi-step:worker 执行第 n 步时调度第 n+1 步,复用 multi-step 代码增量更新模型输入;PP 需特殊处理(末级输出 token 需回传首级)。
- **De-tokenizer 移至 driver 进程**,可视为异步调度的一部分。
- **原生支持多种模型状态**(常规 KV cache、Mamba cache、encoder cache 等),每类缓存有专属内存管理器与 block table。
- **从引擎中移除 beam search**(在引擎外模拟);**prefix-caching 成为一等公民**,用它实现 parallel sampling,移除 SequenceGroup 概念;最小化 `PyObjectCache`。

## 优化点与收益
正文未给出量化数字。动机中的经验教训指明收益来源:GPU 越来越快而 CPU 不会、每步调度需遍历整个 running 队列(纯 Python)、输入广播昂贵、大 batch 下 de-tokenization 开销显著、sampler "pythonize" 输出昂贵——新架构通过有状态 worker、缓存并增量构建输入、异步调度与 de-tokenization 下放,降低 CPU 侧开销、最小化 GPU 空闲时间。

## 现状与落地
仍在推进。定位为透明跟踪 issue,设计由核心维护者(@WoosukKwon、@zhuohan123、@youkaichao、@simon-mo 等)主导,计划按增量方式执行,时间线中 #8726 已勾选;整体落地进度需查关联 PR。

## 关联
- [#8726](https://github.com/vllm-project/vllm/issues/8726)(时间线中已勾选的关联项)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
