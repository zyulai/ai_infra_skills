# RFC #18633: [RFC]: [Spec Decode] Combine Ngram and EAGLE

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18633](https://github.com/vllm-project/vllm/issues/18633) |
| 作者 | ekagra-ranjan |
| 创建日期 | 2025-05-23 |
| 状态 | ⚪ closed(关闭于 2025-09-25) |
| labels | RFC, stale |
| 热度 | 👍 7 · 💬 6 |
| 分类 | Speculative Decoding |

## 一句话结论
提议把 Ngram 与 EAGLE 两种投机解码草稿策略合并为一个 `ngram-eagle` 方法,按请求按时间步自适应选择,免去部署两套实例。

## 提案动机
EAGLE 与 Ngram 在不同场景下各有加速:Ngram 在 prompt 中存在可用 ngram(可据此构造草稿)时受益,如编辑类任务;EAGLE 则是任何通用场景都能提草稿的策略。当前 `vllm serve` 启动时只能在 Ngram 或 EAGLE 中二选一,必须提前决定草稿策略;要在生产中同时支持编辑类请求与一般请求,就得部署不同的模型实例。由于编辑类与非编辑类流量占比并非静态,这会导致 GPU 资源利用不达标。

## 提出的方案
- 合并 Ngram 与 EAGLE 后无需分别部署,实例可在任意时间步为任意请求选择策略。
- 每个 batch 中每条 seq 的草稿合并策略:先查是否存在 ngram 匹配;有则用 ngram 草稿 token,该 seq 跳过 EAGLE;没有则将其加入待 EAGLE 处理的 batch;有需要时运行 EAGLE;最后照常由 target model 与 Rejection Sampler 验证草稿 token。
- 关键依据:实践中 EAGLE 的最优 K 约为 3(即每次最多提 3 个草稿 token,来源为 issue #17812 与 NVIDIA 博客);Ngram 通常为 5 且可设更高。若 `prompt_lookup_max`/`prompt_lookup_min` 设置得当,ngram 命中即说明当前序列/数据在该步大概率适合 ngram lookup;未命中则交给 EAGLE 出草稿。
- 合并策略以新方法名 `ngram-eagle` 提供,不改变 `ngram` 或 `eagle` 的默认行为。
- 开销论证:Ngram 是 model-free drafter,EAGLE 需要草稿模型,合并后运行开销不会比单独 EAGLE 高太多。

## 优化点与收益
正文未给出端到端量化数字,仅引用来源说明「EAGLE 最优 K 为 ~3」;预期收益是相比只用 EAGLE 可提升整体 AL(接受长度),并消除为编辑/非编辑流量分别部署实例带来的 GPU 资源浪费。

## 现状与落地
已关闭(2025-09-25)。作者指出的已知挑战:验证阶段每条 seq 的草稿数不同(EAGLE K≈3,Ngram K≥5),对 torch.compile 与 cugraph 的影响待观察;关闭后的落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/17812
- https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
