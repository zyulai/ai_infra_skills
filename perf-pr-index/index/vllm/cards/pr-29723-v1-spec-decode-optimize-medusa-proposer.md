# PR #29723: [V1][Spec Decode] Optimize Medusa proposer to avoid GPU-CPU sync

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @dongbo910220 |
| 合并 | 2025-12-10 |
| 分类 | Speculative Decoding(tier: major) |
| Label | speculative-decoding, ready, v1 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/29723 |

## 一句话结论

Medusa proposer 的草稿 token 不再转 CPU list,全程留在 GPU,消除每步 decode 的 GPU-CPU 同步,吞吐约 +1%。

## 动机

V1 的 Medusa proposer 在每个调度步把各 Medusa head 的 logits 逐个 `argmax().tolist()` 拉回 CPU,再用 Python list `zip` 转置成 per-request 草稿序列。这带来三类开销:每批草稿 token 的 GPU-to-CPU 拷贝、Python list 操作与转置、以及 `tolist()` 强制的 GPU-CPU 同步点——同步会让 GPU 流水线在每步 spec decode 上停等,batch 越大、token 生成率越高损失越明显。代码里本就留有 TODO 标注应返回 GPU tensor。

## 优化点

机制是改掉 proposer 的返回表示:`propose()` 签名从 `list[list[int]]` 改为 `torch.Tensor`。每个 head 的 logits 仍各自 `argmax(dim=-1)`,但结果用 `torch.stack(..., dim=1)` 在 GPU 上拼成 `[batch_size, num_heads]` 单张量直接返回,下游按该张量消费,`tolist()` 与 Python 转置全部移除。这样草稿 token 在整个 speculative decoding 管线内保持 GPU 驻留,每步零 host 同步。改动极小(单文件、替换 4 行),属于纯数据表示优化,不触碰 Medusa head 模型与验证逻辑。

## 关键改动

```text
vllm/v1/spec_decode/medusa.py: propose() 返回 GPU tensor 而非 CPU 嵌套 list
```

```python
# Shape: [batch_size, num_heads]
draft_tokens = torch.stack(
    [logit.argmax(dim=-1) for logit in logits], dim=1)
return draft_tokens
```

## 性能声明

Mistral-7B-Instruct-v0.2 + Medusa heads,单 GPU(GPU 型号未注明),batch_size=32、max_tokens=256、64 条 diverse prompts,按 git checkout 做 A/B:旧版(`.tolist()` 同步)1184.2 tok/s,优化版(GPU tensor)1195.0 tok/s,约 **1% 吞吐提升**。PR 称该收益随更大 batch 与更高 token 生成率放大。

## 验证与回归风险

PR 正文的 Test Plan 与 Test Result 章节为空,未列出新增/依赖测试。风险点:`propose()` 返回类型变更属于 proposer 接口契约变化,V1 中其他 proposer(Eagle/Ngram 等)与 Medusa 的消费端须按新类型处理;若下游仍有按 list 索引草稿 token 的路径会破坏正确性;收益在大 batch 下更明显,小 batch 场景接近噪声(1% 量级)。

## 关联

同分类(V1 spec decode 去 CPU 同步/采样优化):#13365(n-gram matching 优化)、#14930(优化 rejection sampling)、#25948/#29710(同类避免同步/拷贝的 perf 系列)。Medusa 相关的 spec decode 基础设施见 #22872(Eagle+Medusa 多模态)。
