# Perf PR 索引统计总览

- 生成时间: 2026-11-08T07:30:42+00:00

## 总量

| 仓库 | 收录总数 | 🔥 重点 | 一般 | 有加速声明 |
| --- | --- | --- | --- | --- |
| vllm | 6849 | 936 | 5913 | 535 |
| sglang | 6059 | 645 | 5414 | 522 |

## vllm:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 906 | 182 |
| MoE(混合专家) | 693 | 126 |
| Quantization(量化) | 978 | 159 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 309 | 47 |
| Overlap / Scheduler(重叠与调度) | 331 | 25 |
| Attention / Kernel(注意力与算子) | 1338 | 228 |
| KV Cache / Memory(KV Cache 与显存) | 402 | 37 |
| Parallelism / Communication(并行与通信) | 28 | 7 |
| Sampler / Detokenization(采样与反分词) | 58 | 6 |
| Frontend / Serving(前端与服务) | 72 | 10 |
| Model-specific Fast Path(模型专属快路径) | 323 | 67 |
| CI / Test-Perf(CI 与测试性能) | 569 | 0 |
| Other(其他) | 842 | 42 |

## vllm:年度分布

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 4440 | 675 |
| 2025 | 1933 | 213 |
| 2024 | 430 | 45 |
| 2023 | 46 | 3 |

## vllm:Top 作者 (前 15)

| 作者 | PR 数 |
| --- | --- |
| [mgoin](https://github.com/mgoin) | 249 |
| [yewentao256](https://github.com/yewentao256) | 196 |
| [WoosukKwon](https://github.com/WoosukKwon) | 185 |
| [NickLucche](https://github.com/NickLucche) | 131 |
| [njhill](https://github.com/njhill) | 122 |
| [jeejeelee](https://github.com/jeejeelee) | 102 |
| [LucasWilkinson](https://github.com/LucasWilkinson) | 102 |
| [AndreasKaratzas](https://github.com/AndreasKaratzas) | 101 |
| [DarkLight1337](https://github.com/DarkLight1337) | 100 |
| [MatthewBonanni](https://github.com/MatthewBonanni) | 84 |
| [hmellor](https://github.com/hmellor) | 73 |
| [tlrmchlsmth](https://github.com/tlrmchlsmth) | 70 |
| [Isotr0py](https://github.com/Isotr0py) | 63 |
| [ZJY0516](https://github.com/ZJY0516) | 54 |
| [robertgshaw2-redhat](https://github.com/robertgshaw2-redhat) | 53 |

## vllm:加速声明 Top (数字最大前 20)

| 声明 | PR | 标题 | 声明原文 |
| --- | --- | --- | --- |
| 197.7% | [#28161](https://github.com/vllm-project/vllm/pull/28161) | Adding a benchmark for batch invariance | 197.70% Throughput |
| 169% | [#65761](https://github.com/vllm-project/vllm/pull/65761) | [Bugfix][Multimodal] Encoder CUDA graph: cap each graph's f… | up to 169% |
| 133.7% | [#52861](https://github.com/vllm-project/vllm/pull/52861) | [Model][NVIDIA] Route DSA models to the CUDA non-compiled p… | 133.7%)** and decode-only throughput |
| 132.49% | [#28161](https://github.com/vllm-project/vllm/pull/28161) | Adding a benchmark for batch invariance | 132.49% Throughput |
| 116% | [#46340](https://github.com/vllm-project/vllm/pull/46340) | [Kernel] TD operand loads for batched MoE GEMM (moe_mmk) on… | 116% throughput |
| 103% | [#61458](https://github.com/vllm-project/vllm/pull/61458) | [Perf][Model] Batch PaddleOCR-VL vision encoder and project… | 103% of one core in both versions, so the throughput |
| 100% | [#7957](https://github.com/vllm-project/vllm/pull/7957) | [Frontend] Minor optimizations to zmq decoupled front-end | 100% throughput |
| 100% | [#28603](https://github.com/vllm-project/vllm/pull/28603) | [BugFix] Fix type error when assign a trition kernel tensor… | 100.00% ==Throughput |
| 100% | [#38479](https://github.com/vllm-project/vllm/pull/38479) | [Attention Backend] TurboQuant: 2-bit KV cache compression … | 100%** of baseline throughput |
| 100% | [#57057](https://github.com/vllm-project/vllm/pull/57057) | [Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D… | 100%, evicting)** and throughput |
| 99% | [#18844](https://github.com/vllm-project/vllm/pull/18844) | [Perf] Tune `scaled_fp8_quant` by increasing vectorization | up to 99% |
| 99% | [#52162](https://github.com/vllm-project/vllm/pull/52162) | [Perf][PCP] Shard decode requests across PCP ranks | 99% hit rates)</b></summary>  **Throughput |
| 96% | [#38479](https://github.com/vllm-project/vllm/pull/38479) | [Attention Backend] TurboQuant: 2-bit KV cache compression … | 96%** of baseline - **k8v4 TPOT is faster |
| 95% | [#51473](https://github.com/vllm-project/vllm/pull/51473) | [ROCm][DSV4] Preserve native MXFP4 TP8 shard allocation | 95% throughput |
| 95% | [#52162](https://github.com/vllm-project/vllm/pull/52162) | [Perf][PCP] Shard decode requests across PCP ranks | throughput improves at 95% |
| 95% | [#52162](https://github.com/vllm-project/vllm/pull/52162) | [Perf][PCP] Shard decode requests across PCP ranks | 95%, sharding raises throughput |
| 94.04% | [#60695](https://github.com/vllm-project/vllm/pull/60695) | [Bugfix][Mooncake] Save the EAGLE attention proof with hybr… | 94.04% and throughput |
| 93% | [#45739](https://github.com/vllm-project/vllm/pull/45739) | [Perf] Restore zero-init of swizzled NVFP4 scale buffer to … | 93% drop in output-token throughput |
| 92.8% | [#50294](https://github.com/vllm-project/vllm/pull/50294) | [Kernel][Model] Optimize FA4 mm_prefix range lookup | 92.8% QPS improvement |
| 90% | [#52162](https://github.com/vllm-project/vllm/pull/52162) | [Perf][PCP] Shard decode requests across PCP ranks | throughput improve at every 90% |

> 注: 声明数字由正则从 PR 正文提取, 未经核实; 请点击 PR 查看测试条件 (模型/硬件/workload)。


## sglang:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 881 | 124 |
| MoE(混合专家) | 672 | 89 |
| Quantization(量化) | 665 | 91 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 476 | 43 |
| Overlap / Scheduler(重叠与调度) | 208 | 21 |
| Attention / Kernel(注意力与算子) | 1635 | 157 |
| KV Cache / Memory(KV Cache 与显存) | 116 | 13 |
| Parallelism / Communication(并行与通信) | 46 | 9 |
| Sampler / Detokenization(采样与反分词) | 53 | 5 |
| Frontend / Serving(前端与服务) | 18 | 0 |
| Model-specific Fast Path(模型专属快路径) | 338 | 59 |
| CI / Test-Perf(CI 与测试性能) | 365 | 0 |
| Other(其他) | 586 | 34 |

## sglang:年度分布

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 4273 | 513 |
| 2025 | 1618 | 124 |
| 2024 | 168 | 8 |

## sglang:Top 作者 (前 15)

| 作者 | PR 数 |
| --- | --- |
| [BBuf](https://github.com/BBuf) | 418 |
| [hnyls2002](https://github.com/hnyls2002) | 294 |
| [zhyncs](https://github.com/zhyncs) | 243 |
| [mickqian](https://github.com/mickqian) | 232 |
| [merrymercy](https://github.com/merrymercy) | 201 |
| [ch-wan](https://github.com/ch-wan) | 169 |
| [Fridge003](https://github.com/Fridge003) | 166 |
| [fzyzcjy](https://github.com/fzyzcjy) | 109 |
| [mmangkad](https://github.com/mmangkad) | 107 |
| [b8zhong](https://github.com/b8zhong) | 101 |
| [yuan-luo](https://github.com/yuan-luo) | 97 |
| [Qiaolin-Yu](https://github.com/Qiaolin-Yu) | 90 |
| [ispobock](https://github.com/ispobock) | 89 |
| [DarkSharpness](https://github.com/DarkSharpness) | 88 |
| [kpham-sgl](https://github.com/kpham-sgl) | 80 |

## sglang:加速声明 Top (数字最大前 20)

| 声明 | PR | 标题 | 声明原文 |
| --- | --- | --- | --- |
| 168% | [#34647](https://github.com/sgl-project/sglang/pull/34647) | [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh… | 94%-168% improvement |
| 160.7% | [#33474](https://github.com/sgl-project/sglang/pull/33474) | Select DeepGEMM standard layouts by memory budget | 160.7%** equivalent throughput |
| 160.7% | [#33474](https://github.com/sgl-project/sglang/pull/33474) | Select DeepGEMM standard layouts by memory budget | 160.7%)</strong></summary>  Masked was faster |
| 143.4% | [#43363](https://github.com/sgl-project/sglang/pull/43363) | [Perf] moe: Extend FlashInfer CUTLASS MoE autotune coverage… | 143.4% higher |
| 100% | [#41533](https://github.com/sgl-project/sglang/pull/41533) | [ROCm] Use the fused MLA absorb + RoPE + KV-write kernel fo… | 100% means this PR is better: throughput |
| 98% | [#41830](https://github.com/sgl-project/sglang/pull/41830) | [diffusion] kernels: take the Wan VAE decoder's remaining f… | up to 98% |
| 97% | [#44322](https://github.com/sgl-project/sglang/pull/44322) | [Feat] [AMD] dsa: Enable DCP on the triton sparse-MLA backe… | 97%): throughput |
| 96.8% | [#14133](https://github.com/sgl-project/sglang/pull/14133) | Opt moe align block size kernel | Up to 96.8% |
| 95.7% | [#37778](https://github.com/sgl-project/sglang/pull/37778) | [AMD][DSV4] Enable hicache on deepseek-v4 fp8 unified attn | 95.7%   Throughput |
| 94% | [#34647](https://github.com/sgl-project/sglang/pull/34647) | [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh… | 94%-168% improvement |
| 85% | [#13731](https://github.com/sgl-project/sglang/pull/13731) | [sgl-kernel][Feat][B200][1/N]Support MXFP8 Grouped GEMM in … | up to 85% |
| 82.37% | [#34619](https://github.com/sgl-project/sglang/pull/34619) | [Diffusion][GLM-Image] Retune QK head LayerNorm for SM103 | 82.37% memory throughput |
| 82% | [#47180](https://github.com/sgl-project/sglang/pull/47180) | [Perf] [AMD] dsa: Replicate the DCP query projection on the… | 82%), so P90 stays low; throughput |
| 80.58% | [#35981](https://github.com/sgl-project/sglang/pull/35981) | [diffusion] Flatten Wan VAE RMSNorm row addressing | 80.58% SM throughput |
| 80% | [#36607](https://github.com/sgl-project/sglang/pull/36607) | [AMD] Enable GLM-5.3-Flash on gfx942 and gfx950 | 80% of B200 for total throughput |
| 76% | [#32813](https://github.com/sgl-project/sglang/pull/32813) | [Kernel] Fuse KV-cache writes for asymmetric K/V (head_dim … | 76%, output throughput |
| 75.8% | [#12240](https://github.com/sgl-project/sglang/pull/12240) | [VLM] Optimize qwen_vl preprocess_video | 75.8%)  e2e speedup |
| 75% | [#10321](https://github.com/sgl-project/sglang/pull/10321) | [1/2][sgl-kernel] Support moe_sum_reduce cuda kernel | 75% throughput |
| 75% | [#14640](https://github.com/sgl-project/sglang/pull/14640) | [sgl-kernel][Feat][B200][2/N] Support MXFP8 Grouped GEMM in… | E2E memory bandwidth utilization increased from 75% |
| 75% | [#18124](https://github.com/sgl-project/sglang/pull/18124) | [Perf] Use safetensors `load_file` in multithread loader | speedup that reduced the overall wait time by 75% |

> 注: 声明数字由正则从 PR 正文提取, 未经核实; 请点击 PR 查看测试条件 (模型/硬件/workload)。


## 数据快照(下次更新的基线)

- 快照记录时间: 2026-11-07T12:17:51+00:00
- **vllm** @c9e4cb2b793b (upstream HEAD 2026-11-07) — commit: https://github.com/vllm-project/vllm/commit/c9e4cb2b793b5fa6672071675ef5aa49f7926f7f | API 抓取窗口: 2026-11-07T02:54:22+00:00 ~ 2026-11-07T03:15:12+00:00 | raw 6849 → 重点档 936 → 证据卡 936
- **sglang** @8920463c90c8 (upstream HEAD 2026-11-07) — commit: https://github.com/sgl-project/sglang/commit/8920463c90c81127d9413f093240aeba17983f3a | API 抓取窗口: 2026-11-07T02:50:25+00:00 ~ 2026-11-07T03:28:51+00:00 | raw 6059 → 重点档 645 → 证据卡 645
