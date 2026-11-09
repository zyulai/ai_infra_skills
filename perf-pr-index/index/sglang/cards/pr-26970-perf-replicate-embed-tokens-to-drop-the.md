# PR #26970: [perf] Replicate embed_tokens to drop the post-embed all-reduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Qiaolin-Yu |
| 合并 | 2026-06-02 |
| 分类 | Parallelism / Communication(major) |
| Label | high priority, deepseek, run-ci, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 4 个文件:environ.py、vocab_parallel_embedding.py、deepseek_nextn.py、deepseek_v2.py、kimi_k25_eagle3.py |
| 链接 | https://github.com/sgl-project/sglang/pull/26970 |

## 一句话结论

DeepSeek-V2 系列可把 embed_tokens 在 TP rank 间复制而非按 vocab 分片,免去 embed 后 all-reduce,e2e decode 提升 1–2%,代价是显存中多存整份 embedding。

## 动机

TP 部署下 embedding 按 vocab 维分片时,embed lookup 之后需要一次 all-reduce/all-gather 来合并各 rank 的部分结果。对 embedding 表不太大、decode 延迟敏感的模型,这次通信是可消除的开销;用整表复制的显存换掉一次集合通信是显式 trade-off。

## 优化点

新增 `SGLANG_ENABLE_EMBED_REPLICATION` 开关(默认 False)与 `get_embedding_tp_kwargs()` 统一入口:开启时返回 `{"enable_tp": False}`,即在每个 rank 上复制完整 embedding 表,跳过 embed 后的 all-reduce;关闭时维持原状(`enable_tp=True` 按 vocab 分片,DP attention 下各 rank 只拥有本地 token,在 attention-TP group 内 reduce 而非全 TP group)。关键约束:EAGLE/NextN draft 与 target 共享 `embed_tokens.weight` 张量(`set_embed`/`set_embed_and_head`),target 与每个共享该 embedding 的 draft 必须使用同一 vocab-parallel layout,否则 draft 的 masking/index 数学作用在不同布局的张量上,accept_len 会静默下降——因此 Deepseek-V2 家族(DeepSeek V3.1、Kimi K2.5)target 与其 EAGLE3/NextN draft(deepseek_nextn.py、kimi_k25_eagle3.py)全部改走同一 helper,避免布局漂移。

## 关键改动

```text
python/sglang/srt/environ.py: 新增 SGLANG_ENABLE_EMBED_REPLICATION(默认 False)
python/sglang/srt/layers/vocab_parallel_embedding.py: 新增 get_embedding_tp_kwargs,
  统一 target 与 draft 的 embedding vocab-parallel 布局
python/sglang/srt/models/deepseek_v2.py: embed_tokens 构造改用 get_embedding_tp_kwargs
python/sglang/srt/models/deepseek_nextn.py: 同上(NextN draft)
python/sglang/srt/models/kimi_k25_eagle3.py: 同上(EAGLE3 draft)
```

关键摘录(helper 逻辑):

```python
if envs.SGLANG_ENABLE_EMBED_REPLICATION.get():
    # Replicate the full table on every rank: skips the embed all-reduce
    return {"enable_tp": False}
return {"enable_tp": True, "use_attn_tp_group": is_dp_attention_enabled()}
```

## 性能声明

PR 正文仅一句:"1% ~ 2% e2e decoding performance improvement, it's a trade-off since it needs to store the total embedding layer now."。未注明模型、硬件、TP 配置、workload 等测试条件。

## 验证与回归风险

PR 正文的 Accuracy/Modifications/Motivation 章节均为空,未提供精度测试或 profiling 证据;checklist 全部未勾选。回归风险:(1) 默认关闭,需显式开 env;(2) target 与共享 embedding 的 draft 布局必须一致——PR 用单一 helper 强制同步,注释明确说明若不一致 accept_len 会静默下降;(3) 显存代价:每 rank 多存整份 embedding;(4) 目前仅适用于 DeepSeek-V2 家族(DeepSeek V3.1、Kimi K2.5)及其 draft,其他模型不受影响。

## 关联

无(pack 内未提及直接关联 PR;属 DeepSeek 系列通信优化方向,label 含 deepseek)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
