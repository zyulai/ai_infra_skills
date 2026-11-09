# RFC #48202: [RFC]: Per-request effective proposal lengths for adaptive speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#48202](https://github.com/vllm-project/vllm/issues/48202) |
| 作者 | LJX-xixi |
| 创建日期 | 2026-07-10 |
| 状态 | ⚪ closed(关闭于 2026-08-11) |
| labels | RFC |
| 热度 | 👍 0 · 💬 13 |
| 分类 | Speculative Decoding |

## 一句话结论
提议为投机解码增加按请求的 proposal_lengths 元数据:物理 buffer 保持固定形状,逻辑上只取每请求有效前缀参与验证与记账,为自适应策略提供接口契约。

## 提案动机
vLLM 已支持 batch 统一的投机深度自适应:Dynamic SD(#32374)离线分析位置级接受率与 ITL、运行时按 goodput 选择每步统一 K(当前 EAGLE-1),#45953 扩展到 full CUDA graphs;#36657/#35301 提出置信度阈值早退,但刻意保持 batch 统一——输出固定为 `[batch_size, K]` 加零填充,其 PR 注明「per-request 策略更高效但需要变长草稿输出与更复杂的 KV-cache 记账」。本 RFC 处理的接口问题与上述正交:在同一 batch 内表示不同请求的有效 proposal 长度。实测依据:`deepseek-ai/dspark_qwen3_4b_block7` + Qwen3-4B 上,平均接受长度 GSM8K 为 6.12、MT-Bench 为 3.63(block size 7,与 DSpark 论文 arXiv:2607.05147 Table 1 一致);且单步内逐请求 confident-prefix 长度横跨 0..7 全区间,如 `[7, 4, 0, 7, 2, 1, 0, 0]`。统一 K 无论如何选择,都要在「冷请求的浪费验证」与「热请求的接受损失」之间付出代价。原型(基于 vLLM main `b8c7c8653`)还观察到:修改逐请求 `DraftTokenIds` 在关闭 async scheduling 时能改变有效截断,但在默认 async 路径上不影响验证与 scheduler 记账。

## 提出的方案
- 逻辑表示、物理固定形状:最大投机深度 K 的 decode step 中,proposer 可返回固定形状 `[R, K]` 的 `draft_token_ids`(同现状)加上可选 int32 向量 `proposal_lengths`(`[R]`,`0 <= proposal_lengths[r] <= K`)。
- 语义:对请求 r,只有前缀 `draft_token_ids[r, :proposal_lengths[r]]` 语义上存在;该前缀之外的位置不得参与验证、KV-cache 记账、output-placeholder 记账与投机解码指标;`proposal_lengths` 缺省时逐位保持现有 uniform-K 行为。
- 长度向量绑定产生它的那个 in-flight batch;async 路径下不得追溯修改其他 batch 的记账。后端是物理紧凑打包还是保留填充属实现选择,接口只定义逻辑长度与记账不变量——即把「变长草稿输出」重构为「固定形状 buffer + 逐请求逻辑长度」。
- 待维护者决策的设计问题:该元数据是扩展 #32374 Dynamic SD 的现有机制,还是作为 proposer/model-runner 输出上的独立字段由 scheduler 记账消费;作者已有第二种形态的可工作原型,开 PR 前会收敛到任一偏好的边界。
- 分三步的 PR 计划:1) `[Core][Spec Decode]` 逐请求 proposal-length 元数据,默认行为不变,sync/async 路径均测试,不附策略;2) `[Model][Spec Decode]` 接入 Qwen3 DSpark confidence head 作为首个消费者(DeepSeek 发布 checkpoint 已带该 head,vLLM 加载时跳过,注释为 `confidence_head is not wired into inference yet`;head 为单个 `Linear(hidden + markov_rank -> 1)`,每个草稿位置一次轻量线性投影),默认关闭的极小阈值 + 小型 benchmark;3) 策略类工作作为后续单独跟进。
- 原型参考分支:https://github.com/LJX-xixi/vllm/tree/per-request-proposal-lengths-ref(五个聚焦改动 + 单元测试)。

## 优化点与收益
正文明确「不做性能声明」。诚实测量:Qwen3-4B、单张 L20X、batch 最高 64 时,置信度早停在所试任何阈值下都没有比固定满 block 改善 wall clock——DSpark backbone 无论截断与否都一次遍历草稿整个 block,该机制下每步成本对验证 token 数近乎平坦,而损失的接受率以额外解码步数偿还。结论:逐请求截断并非普遍有利,其价值取决于验证算力的机会成本(更大的 target、饱和的引擎或其他自适应策略下可能更高);接口是任何逐请求方法(SpecDec++ 式 head、#35301 明确推迟的逐请求扩展)的前提。实测数字:GSM8K 平均接受长度 6.12、MT-Bench 3.63(dspark_qwen3_4b_block7,Qwen3-4B,block size 7)。

## 现状与落地
已关闭(2026-08-11)。反馈期两周(至 7 月 24 日)或按 spec-decode 维护者偏好;作者表示可立即把参考分支开成 draft PR。最终接口是否合入、采用哪种形态,需查关联 PR。

## 关联
- 参考分支: https://github.com/LJX-xixi/vllm/tree/per-request-proposal-lengths-ref
- 相关:#32374 / #45953(batch 级 Dynamic SD)、#36657 / #35301(batch 均值置信度早退)、RFC #46295(MoE utility budgeting)、#28693 / #26504 / #44885 / #43522(早期自适应尝试)
- DeepSpec: github.com/deepseek-ai/DeepSpec(confidence head 配方)、SpecDec++: arXiv:2405.19715(接受预测 head)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
