# PR #29735: [GDN] Support FlashInfer GDN prefill with extra-buffer radix cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-07-29 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, speculative-decoding, run-ci |
| Diff | diffstat 缺失(素材未提供增删行统计,diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/29735 |

## 一句话结论

让 FlashInfer GDN prefill 支持状态检查点,从而可与 extra_buffer 类 Mamba radix cache 共用,解锁更大 TPS 与更低 TTFT。

## 动机

本 PR 是 #29734 的后续。此前 FlashInfer GDN prefill 无法参与 SGLang 的 Mamba `h`/track-state 流程,因此 `extra_buffer` 与 `extra_buffer_lazy` 两种 radix cache 策略只能回落到 Triton prefill,尽管 FlashInfer 本身具备输出中间状态检查点的能力。这让 SM100/SM103 上更快的 FlashInfer prefill 路径被排除在主流缓存配置之外。

## 优化点

核心机制是让 FlashInfer GDN prefill 按 `mamba_cache_chunk_size` 的间隔输出 packed `state_checkpoints`,并映射到 SGLang 的 tracked-state 元数据:对齐的跟踪边界直接读最终状态池,非对齐边界从 packed checkpoint 取;同时支持非零前缀与 packed 变长 batch。SM100/SM103 上状态与 checkpoint 保持原生 BF16(要求 FlashInfer 0.6.14),SM90 维持 FP32。策略层移除了"extra_buffer 策略不自动选 FlashInfer"的限制,在 #29734 已验证的选择器域内对 `no_buffer`、`extra_buffer`、`extra_buffer_lazy` 自动选择 FlashInfer;Triton tree verify fallback 保留。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/gdn_backend.py: 移除 extra_buffer 策略排除;init_forward_metadata 构建 checkpoint plan
python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py: 新增 maybe_build_flashinfer_checkpoint_plan;extend() 接受 checkpoint 参数并返回 packed h
python/sglang/srt/layers/attention/mamba/mamba2_metadata.py: ForwardMetadata 增加 state_checkpoint_cu_starts/num_state_checkpoints 等字段
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: _track_mamba_state_extend 改为接受 Optional h
test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py: 策略与 checkpoint plan 单测
```

关键片段(checkpoint plan 的非对齐边界映射):

```python
use_checkpoint = track_mask & (relative_track_lens % checkpoint_every_n_tokens != 0)
track_checkpoint_src = checkpoint_cu_starts[:-1][use_checkpoint] + (
    relative_track_lens[use_checkpoint] // checkpoint_every_n_tokens - 1
)
if track_checkpoint_src.numel() and track_checkpoint_src.min() < 0:
    raise ValueError("Tracked GDN state precedes the first FlashInfer checkpoint.")
```

## 性能声明

Checkpoint 路径(InferenceMAX AgentX,Qwen3.5-397B-A17B-NVFP4,disaggregated 1P1D:DEP4 prefill + DEP4 decode,8×GB300,radix + hierarchical cache,concurrency 128,FlashInfer 0.6.14):

| Config | Prefix-cache hit | Total TPS/GPU | TTFT p50 |
|---|---:|---:|---:|
| `extra_buffer` + Triton prefill | 92.1% | 46,161 | 13.5 s |
| `extra_buffer_lazy` + Triton prefill | 92.4% | 47,771 | 13.5 s |
| `extra_buffer_lazy` + FlashInfer prefill | 92.4% | 53,004 | 10.5 s |

即 FlashInfer prefill 相比 Triton 在 `extra_buffer_lazy` 下 TPS/GPU 47,771→53,004,TTFT p50 13.5s→10.5s。准确性:Qwen3-Next-80B-A3B-Instruct TP4 配对 GSM8K,Triton 96.51251% vs FlashInfer 96.73995%(1,319 样本);cache-hit K3 KL 门限 <0.005 双双通过。早期 B200 `extra_buffer_lazy` + overlap 门:prefill/decode cache-hit KL 0.0009038/0.0006804(均 <0.002)。

## 验证与回归风险

新增 GPU 正确性测试 `test_prefill_tracked_state_checkpoints`(FlashInfer vs Triton 输出与 tracked state 双向 assert_close,atol/rtol 3e-2,覆盖非零前缀、packed 变长、非对齐 checkpoint),以及策略单测(三种 radix 策略均选 FlashInfer、显式 override 保留、checkpoint plan 数值断言)。依赖 FlashInfer >= 0.6.14。风险路径:SM90(FP32 状态)与 SM100/SM103(BF16)行为分叉;`mamba_cache_chunk_size` 决定 checkpoint 间隔;非对齐跟踪若早于首个 checkpoint 会抛 ValueError;Triton 仍是 tree verify 的 fallback。

## 关联

前序:#29734(FlashInfer GDN prefill 选择器域)。同区域:GDN/Mamba 线性注意力 prefill 后端策略相关 PR;Mamba radix cache(extra_buffer 系列)优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
