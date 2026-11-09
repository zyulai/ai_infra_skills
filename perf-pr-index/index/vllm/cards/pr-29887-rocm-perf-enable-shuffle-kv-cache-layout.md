# PR #29887: [ROCm][Perf] Enable shuffle kv cache layout and assembly paged attention kernel for `AiterFlashAttentionBackend`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ganyi1996ppo |
| 合并 | 2026-01-15 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/29887 |

## 一句话结论

ROCm 启用 SHUFFLE KV 布局与汇编 paged attention，Qwen3-30B-A3B-FP8 吞吐约 +20%。

## 动机

PR 正文 Purpose 说明为 AiterFlashAttentionBackend 加入 aiter 的 assembly paged attention kernel，并新增 USING_SHUFFLE_LAYOUT 开关控制是否启用，默认设为 False 以防止任何意外情况，待用户充分验证后再移除该 flag。正文未进一步展开改动前的具体瓶颈细节。

## 优化点

新增环境变量 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT（默认 False），vllm/_aiter_ops.py 增加 is_shuffle_kv_cache_enabled。rocm_aiter_fa.py 把 gather kernel 改为按 head 运行（新增 head_id = tl.program_id(1)），支持 SHUFFLE 布局并应用 per-slot fp8 dequant scales；新增 Triton reshape_and_cache_shuffle_* 把 K/V 写入 shuffle 布局并计算/存 fp8 scales，必要时注册 per-block k/v scale buffers；decode 路径改用 aiter.pa_fwd_asm，保留 NHD + paged_attention_v1 作为 fallback，shuffle 下禁用 sliding window，kv_cache_dtype 为 fp8 时启用反量化。以上机制要点来自 PR 正文内的 Cursor Bugbot 摘要与 diff。

## 关键改动

```text
vllm/envs.py: 新增 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT 开关，默认 False
vllm/_aiter_ops.py: 新增 _SHUFFLE_KV_CACHE_ENABLED 与 is_shuffle_kv_cache_enabled
vllm/v1/attention/backends/rocm_aiter_fa.py: gather kernel 按 head、SHUFFLE 布局、fp8 dequant、pa_fwd_asm decode 路径
```

```diff
     VLLM_ROCM_FP8_PADDING: bool = True
     VLLM_ROCM_MOE_PADDING: bool = True
     VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
+    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False
     VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
```

## 性能声明

PR 正文称在 Qwen3-30B-A3B-FP8、MI308 上观察到 "at least about 20% throughput gain and obvious latency reduction on tpot"。vllm bench（200 requests、最大并发 200、总输入 768000 / 生成 204800 tokens）中 normal aiter paged attention 与 assembly aiter paged attention 对比：Request throughput 2.10 → 2.51 req/s，Output token throughput 2149.35 → 2567.23 tok/s，Total Token throughput 10209.43 → 12194.33 tok/s，Mean TPOT 71.65 → 56.62 ms，Median TPOT 72.00 → 56.95 ms，Mean TTFT 21644.01 → 21566.08 ms。测试条件：模型 Qwen3-30B-A3B-FP8，硬件 MI308，精度 FP8，workload vllm bench serve。

## 验证与回归风险

Test Plan 为 gsm8k 精度与 vllm bench 性能。gsm8k 5-shot：normal flexible-extract 0.8264 / strict-match 0.8901，asm 0.8294 / 0.8848。回归风险：新开关默认 False，正文说明待用户完全验证后才移除；shuffle 布局下禁用 sliding window；保留 NHD + paged_attention_v1 fallback；涉及 ROCm Aiter 后端、fp8 KV cache 反量化与 per-head gather 路径。正文内另附 Cursor Bugbot 对 commit 0fa5c519e813e7e42d075da135ff86c47b776246 的自动摘要，复述了 SHUFFLE 布局、per-head gather、fp8 dequant、reshape_and_cache_shuffle_* 与 aiter.pa_fwd_asm decode 路径，以及 shuffle 下禁用 sliding window 等要点；Test Result 段给出 normal 与 assembly 两组完整 serving benchmark 输出和 gsm8k 精度表。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
