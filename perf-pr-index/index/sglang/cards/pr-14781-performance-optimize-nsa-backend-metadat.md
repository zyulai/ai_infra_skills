# PR #14781: [Performance] optimize NSA backend metadata computation for multi-step speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Johnsonms |
| 合并 | 2025-12-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/14781 |

## 一句话结论

NSA 多步投机解码 metadata 改为"一次预计算、多次拷贝",初始化路径提速 1.7x-2.2x。

## 动机

多步投机解码下 CUDA graph replay 需要为 `speculative_num_steps` 个 attn backend 实例各自初始化 metadata:旧路径对每个 backend 重复计算等价 metadata(约 210µs × N),在 4-8 个 speculative steps 时重复开销显著。改为共享 metadata 只算一次、再拷贝/复用(约 210µs + 70µs × N),可大幅压缩 decode 关键路径上的准备时间,且不改变模型计算、投机算法与接受行为。

## 优化点

新增 `PrecomputedMetadata` dataclass 存放共享 metadata(cache_seqlens、cu_seqlens_k、page_indices、real_page_table、nsa_cache_seqlens/nsa_cu_seqlens_k、max_len/max_seqlen_k、可选 FlashMLA metadata);`NativeSparseAttnBackendMTPPrecomputeMixin` 提供 `_precompute_replay_metadata`,按 decode / target_verify / draft_extend 三种 forward mode 分别预计算(target_verify 需 `seq_lens + speculative_num_draft_tokens` 并对 page table 做 `repeat_interleave`);`nsa_backend.py` 新增 `init_forward_metadata_replay_cuda_graph_from_precomputed` 纯拷贝快速路径,replay 时先对 `attn_backends[0]` 预计算一次,再循环 `copy_` 到各后端。由环境变量 `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA`(默认 true)控制,关闭可回退旧路径。

## 关键改动

```text
python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py: PrecomputedMetadata 与各 mode 预计算(新文件)
python/sglang/srt/layers/attention/nsa_backend.py: precomputed 拷贝式 replay 初始化 + mixin 接入
python/sglang/srt/layers/attention/nsa/utils.py: SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA 开关
```

## 性能声明

metadata 计算 642µs → 437µs(31.9% faster),kernel launch 60 → 40(33.3% fewer);metadata 初始化路径在 4-8 speculative steps 下 1.7x-2.2x faster。端到端:单请求 16K 序列 +12.4% TPS;4 并发、16K-64K 序列 +5.5% ~ +7.5% TPS;TTFT 影响约 ±3% 内;E2E benchmark 约 +5% ~ +15% TPS。模型与硬件条件 PR 未注明。

## 验证与回归风险

PR 正文声明仅改 metadata 准备,不改投机解码算法、attention 语义或接受行为;env 开关保留旧路径 fallback;拷贝路径按 mode 分支覆盖 decode/target_verify/draft_extend 的 page_table 与 NSA seqlens 切片拷贝,backend 元数据布局若变化需同步;PR 未列新增单测。

## 关联

后续 PR #17554(同一 NSA 多步投机解码 metadata 路径,进一步用 fused CUDA kernel 消除拷贝的 launch 开销)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
