# PR #57640: [ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rbrugaro-amd |
| 合并 | 2026-09-30 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/57640 |

## 一句话结论

ROCm 把 Kimi-K3 MLA decode 的 KV-cache 写入与 Q-prep 合成一次 AITER launch，MI355X TP8 下 kernel 时间 1.98×。

## 动机

ROCm 上 Kimi-K3 的 MLA decode 把每层 query preparation 与 KV-cache 写入拆成两次 launch：`vllm::concat_and_cache_mla_kernel`（fp8 KV-cache 写入）与 `triton_poi_fused__to_copy_cat_clamp_mul_reciprocal_view_0`（`[ql_nope | q_pe]` concat 与该 query 的静态 fp8 量化）。本 PR 旨在把两者合并为一次 AITER launch，减少 kernel 数量与 device time。

## 优化点

合成单个 AITER `fused_qk_rope_concat_and_cache_mla` launch。Kimi-K3 MLA 是 NoPE（`mla_use_nope`），该 kernel 却总应用 RoPE 并要求 cos/sin cache，故喂恒等旋转（`cos = 1, sin = 0`）使旋转退化为空操作（与 ROCm/ATOM 路由 K3 的做法相同）；恒等 cache 仅一行，依赖 AITER per-head / `_opt` decode kernel 对 `pos` 的 clamp，因此拒绝会选中不 clamp 的 general decode kernel 的头数（`kv_lora_rank * num_heads < 2048`）。全部改动落在单文件 `vllm/models/kimi_k3/amd/mla.py`：`_fused_qk_prep_supported()` 在 `__init__` 一次性判定平台、KV-cache dtype、形状、DCP/PCP/HiSparse、head padding 与 W_UK 变体；`_fused_decode()` 复刻 `MLAAttention.forward_impl` 的 W_UK 分派优先级（fp4 先于 fp8）保证 `ql_nope` bit-identical，再继续 `impl.forward_mqa` 与 `_v_up_proj`。`forward()` 只剩"是否纯 decode batch"一个运行时条件，其余走原路径；正文强调不碰 shared code、零 blast radius。

## 关键改动

```text
vllm/models/kimi_k3/amd/mla.py: _fused_qk_prep_supported 在 __init__ 内一次性完成静态 gating
vllm/models/kimi_k3/amd/mla.py: _fused_decode 复刻 W_UK 分派并发出 AITER fused_qk_rope_concat_and_cache_mla 调用
```

## 性能声明

MI355X、TP=8、Kimi-K3 fp4、`VLLM_ROCM_USE_AITER=1`（唯一环境变量），基于 stock upstream nightly `vllm/vllm-openai-rocm:nightly-rocm100-0bfc7a15d095fe83ecc82b50561a93c177fece2d`（2026-09-16）、8× MI355X，无 vendor patch。Rank-0 torch profile、6 engine steps、24 MLA layers：两个被替换 kernel 各 144 次（668.0 / 683.8 µs）归零，新 `aiter::fuse_qk_rope_concat_and_cache_mla_kernel` 144 次（682.8 µs），即 288 → 144 launches、1351.8 → 682.8 µs = 1.98×，每 MLA layer 每 decode step 9.39 → 4.74 µs。GSM8K（1319 题、5-shot、temperature=0、top_p=1、max_tokens=4096、chat template）：fused strict-match 0.9545、flexible-extract 0.9553、stderr 0.0057。端到端（ISL 1024 / OSL 8192、C=1|8|32）：Conc 1/8 的 TPOT p50、ITL p50/p90、out tok/s 在 0.2% 内平坦，Conc 32 约向 baseline 偏 0.7%，正文结论"serving throughput improvement is below 1%"，并注明 `cross_device_reduce_2stage` 单独约占 rank-0 decode device time 46%。

## 验证与回归风险

Test Plan 给出 serve、`vllm bench serve`（random 1024 in / 8192 out）、torch profiler kernel census 与 lm_eval gsm8k 命令，数字来自无 vendor patch 的 stock nightly。作者强调零 blast radius：不改 shared `layers/mla.py`（不同于堆叠 #47757 的 #57452），仅动 `vllm/models/kimi_k3/amd/`，与 #47757 无依赖、可任意顺序合并。风险路径由 `_fused_qk_prep_supported()` 显式回退：混合 batch、DCP/PCP/HiSparse、fp8-bmm/AMX W_UK、head padding、非 ROCm、非 fp8 KV cache、会选中 non-clamping general decode kernel 的头数等。正文未给独立回归测试计数。

## 关联

正文明确列出：#57452（同一优化，本 PR supersedes 并将关闭）、#47757（sparse MLA / deepseek_v32，互补）、#55230（DeepSeek-R1，改 shared `layers/mla.py`）、#45798 / #38313 / #41099 / #41101（同一融合的 torch.compile graph-rewrite，K3 非 `@support_torch_compile` 故不触发）、#53166（K3 ROCm chunked-context gather）、#52494（已合并，同 wrapper 的 q/kv RMSNorm 融合）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
