# RFC #33118: [RFC]: Hidden States Extraction

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33118](https://github.com/vllm-project/vllm/issues/33118) |
| 作者 | fynnsu |
| 创建日期 | 2026-01-26 |
| 状态 | ⚪ closed (2026-03-03) |
| labels | RFC |
| 热度 | 👍 17 · 💬 18 |
| 分类 | Other |

## 一句话结论
复用 KV cache 体系从任意模型层提取 hidden states,为 Eagle3 等投机解码草稿模型训练提供高性能数据通路。

## 提案动机
Eagle3、DFlash 等投机解码算法以 verifier 模型多层的 hidden states 为输入,训练草稿模型需要带对应 token 的 hidden states 数据集,而 vLLM 目前没有官方提取途径;hidden states 访问对 RL(如 VERL)与量化研究同样有价值。现有做法两类:改用 transformers(丢失 vLLM 性能优化与分布式支持,且与 vLLM hidden states 的细微不一致易生 bug)或重度 patch vLLM(维护负担大,常需禁用 prefix caching、auto batching、async server 等)。此前尝试(#15434、#24288、#26686)或拖累 hotpath、或仅限末层/pooling 模型。既定要求:关闭时零性能成本、最小复杂度、内存管理、进程间非阻塞高效传输。

## 提出的方案
- 关键洞察:① 已有 `aux_hidden_states` 机制(Eagle3/DFlash 推理在用);② KV Connector 是可扩展公共 API;③ hidden states 与 token 序列 1-to-1 对应、前缀可像 KV cache 一样复用;④ hidden states 尺寸与 K/V 状态相当或数倍,且 vLLM 支持跨层非统一 KV 页大小。
- 方案:构造一个 dummy Eagle3 模型,接收 hidden states 并存入 dummy attention 层的 KV cache,再经 KV Connector 接口高效提取;dummy 模型返回随机 token 预测(如 token 0),会被 verifier 拒绝。
- 已有可运行的树外 PoC(fynnsu/vllm-hidden-states-extractor),用 vLLM 插件系统注册组件,完全不改动 vLLM;RFC 目的为征集方案反馈并确定进 tree 的路径。
- 层选择:经 `eagle_aux_hidden_state_layer_ids` 配置任选层与层数;Eagle3 训练通常需 4 层 hidden states(3 层作模型输入、1 层重算目标分布),PoC 当前即提取 4 层。
- shape 不匹配问题由 #25101(允许各层不同 hidden 维度)解决,把 hidden states 存入 KV cache 也让 vLLM 在分配 block 时计入其存储。

## 优化点与收益
正文未给出量化数字。定性收益:关闭时零性能开销;保留 async server、chunked prefill、continuous batching、prefix caching 等 vLLM 特性;避免两步流程——可在生成新文本的同时返回 hidden states,改善"用 verifier 重新生成数据集响应以提升草稿模型接受率"的研究流程。

## 现状与落地
已关闭(2026-03-03)。正文自述有可用 PoC(树外),并计划继续以插件形式开发稳定或部分进 tree;最终落地方式正文看不出,需查关联 PR。

## 关联
- [PoC: fynnsu/vllm-hidden-states-extractor](https://github.com/fynnsu/vllm-hidden-states-extractor)
- [PR #15434](https://github.com/vllm-project/vllm/pull/15434)(此前失败尝试)
- [#24288](https://github.com/vllm-project/vllm/issues/24288)(仅末层)
- [PR #26686](https://github.com/vllm-project/vllm/pull/26686)(仅 pooling 模型)
- [PR #25101](https://github.com/vllm-project/vllm/pull/25101)(层异构 hidden 维度)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
