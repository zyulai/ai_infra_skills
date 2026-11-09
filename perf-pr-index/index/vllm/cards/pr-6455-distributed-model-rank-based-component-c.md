# PR #6455: [Distributed][Model] Rank-based Component Creation for Pipeline Parallelism Memory Optimization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wushidonguc |
| 合并 | 2024-07-17 |
| 分类 | KV Cache / Memory |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/6455 |

## 一句话结论

Pipeline Parallel 下按 rank 裁剪模型组件创建(embedding/norm/lm_head/logits/sampler),Llama-3-70B + PP=8 每 rank 显存最高省 25%。

## 动机

改动前,Llama 模型初始化在 Pipeline Parallel 场景下会在每个 rank 上创建全部组件与所有层,即使该 rank 只执行其中一段。这带来不必要的显存开销:中间 rank 不需要 embedding、norm、lm_head、logits processor 与 sampler,却都要占用权重与参数显存,挤压可用于 KV cache / serving 的显存。

## 优化点

按 rank 的 pipeline stage 职责选择性创建组件:embedding 层(`VocabParallelEmbedding`)只在 first rank 创建(`tie_word_embeddings` 时因与 lm_head 共享权重,也需在 last rank 创建);final norm(`RMSNorm`)只在 last rank 创建;`lm_head`、`logits_processor`、`sampler` 只在 last rank 创建。不承担对应职责的 rank 上以 `PPMissingLayer` 占位,保持模型结构完整性(权重加载路径中 `is_pp_missing_parameter` 已有处理)。机制上是把无条件的 `nn.Module` 构造改为按 `get_pp_group().is_first_rank / is_last_rank` 分支,不改变任何前向计算逻辑——各 stage 仍只运行自己持有的层。

## 关键改动

```text
vllm/model_executor/models/llama.py: embed_tokens/norm/lm_head/logits_processor/sampler 改为按 PP rank 条件创建,否则 PPMissingLayer
```

```python
if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                    and get_pp_group().is_last_rank):
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()
```

## 性能声明

PR 正文:`--pipeline-parallel-size 8` 下,Llama-3-70B 每 rank 显存节省最高 25%("up to 25% per-rank memory savings")。吞吐方面仅定性表述:节省的显存可用于 serving,从而可能提升吞吐;并称已进行测量显存节省与性能影响的 benchmark,但具体吞吐数字与测试硬件/精度 PR 未注明。

## 验证与回归风险

PR 正文未列出新增测试(作者称做过显存与性能 benchmark,数字未贴出)。风险面:非 llama 的其他模型不在此 PR 范围内,需后续逐模型适配同样的 rank 条件创建;`tie_word_embeddings` 模型的 embedding/lm_head 权重共享路径依赖 first/last rank 同时创建的分支,PP 非 1/最后 rank 与 LoRA(padded vocab)、量化 lm_head 等组合需要确认权重加载与 sampler 交互;`PPMissingLayer` 占位在 `load_weights` 中的跳过逻辑依赖既有 `is_pp_missing_parameter` 处理。

## 关联

无(本批索引中无其他 Pipeline Parallel 显存优化 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
