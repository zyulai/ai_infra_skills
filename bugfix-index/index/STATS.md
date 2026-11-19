# Bug Fix 索引统计总览

- 生成时间: 2026-11-19T16:27:33+00:00

## 总量

| 仓库 | 收录总数 | 🔥 重点 | 正确性 | 崩溃 | 挂死 | 资源 | 回归 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vllm | 8205 | 663 | 287 | 184 | 92 | 75 | 96 |
| sglang | 6619 | 617 | 307 | 202 | 93 | 77 | 122 |

## vllm:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 295 | 46 |
| MoE(混合专家) | 446 | 49 |
| Quantization(量化) | 568 | 47 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 303 | 28 |
| Overlap / Scheduler(重叠与调度) | 133 | 16 |
| Attention / Kernel(注意力与算子) | 843 | 96 |
| KV Cache / Memory(KV Cache 与显存) | 328 | 29 |
| Parallelism / Communication(并行与通信) | 118 | 9 |
| Sampler / Detokenization(采样与反分词) | 109 | 16 |
| Frontend / Serving(前端与服务) | 513 | 30 |
| Model-specific Fast Path(模型专属快路径) | 724 | 79 |
| CI / Test-Perf(CI 与测试性能) | 26 | 0 |
| Other(其他) | 3799 | 218 |

## vllm:bug 类型分布

| 类型 | 总数 | 🔥 重点 |
| --- | --- | --- |
| correctness(正确性) | 287 | 287 |
| crash(崩溃) | 184 | 87 |
| hang(挂死) | 92 | 33 |
| resource(资源泄漏/OOM) | 75 | 33 |
| regression(回归) | 96 | 95 |
| startup(启动/加载) | 106 | 2 |
| general(一般) | 7365 | 126 |

## vllm:年度分布(按合并时间)

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 4935 | 469 |
| 2025 | 2316 | 152 |
| 2024 | 793 | 38 |
| 2023 | 161 | 4 |

## vllm:热度 Top 20 (reactions×2 + comments)

| 热度 | PR | 标题 | 类型 |
| --- | --- | --- | --- |
| 114 | [54706](https://github.com/vllm-project/vllm/pull/54706) | [ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism | correctness(正确性) |
| 74 | [34789](https://github.com/vllm-project/vllm/pull/34789) | [Bugfix] Offload blocking tokenizer ops to shared thread po… | general(一般) |
| 74 | [35219](https://github.com/vllm-project/vllm/pull/35219) | [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU | general(一般) |
| 64 | [60762](https://github.com/vllm-project/vllm/pull/60762) | [Bugfix][DSv4.1] Declare 64-token kernel blocks for SM120 s… | general(一般) |
| 63 | [44784](https://github.com/vllm-project/vllm/pull/44784) | [Bugfix] Prevent cuMemcpyBatchAsync segfault with MTP and K… | crash(崩溃) |
| 62 | [52921](https://github.com/vllm-project/vllm/pull/52921) | [Bugfix] Align CPU offload pool size across PP/spec-decode … | general(一般) |
| 62 | [57797](https://github.com/vllm-project/vllm/pull/57797) | [Bugfix][RL] Load attention sinks through weight_loader for… | general(一般) |
| 61 | [42143](https://github.com/vllm-project/vllm/pull/42143) | fix(eagle3): read norm_before_fc from eagle_config for NVID… | general(一般) |
| 60 | [56372](https://github.com/vllm-project/vllm/pull/56372) | [Bugfix][Multimodal] Fix flat/scoped mm_processor_kwargs me… | general(一般) |
| 59 | [7874](https://github.com/vllm-project/vllm/pull/7874) | [Bugfix] Fix #7592 vllm 0.5.4 enable_chunked_prefill throug… | general(一般) |
| 57 | [28729](https://github.com/vllm-project/vllm/pull/28729) | [Bugfix] Multiple fixes for gpt-oss Chat Completion prompti… | general(一般) |
| 53 | [44993](https://github.com/vllm-project/vllm/pull/44993) | [Bugfix][Structured Output][Spec Decode] Advance grammar ac… | general(一般) |
| 53 | [57796](https://github.com/vllm-project/vllm/pull/57796) | [Frontend][Multimodal] Accept features together with conten… | general(一般) |
| 52 | [17934](https://github.com/vllm-project/vllm/pull/17934) | Fix CUDA permute/unpermute for use with DeepGemm Moe | general(一般) |
| 50 | [52720](https://github.com/vllm-project/vllm/pull/52720) | [Bugfix] Support MistralCommonBackend tokenizers in structu… | general(一般) |
| 48 | [16618](https://github.com/vllm-project/vllm/pull/16618) | [Bugfix] Fix GLM4 model | general(一般) |
| 48 | [44297](https://github.com/vllm-project/vllm/pull/44297) | [Bugfix][Structured Output][Spec Decode] Constrain bitmask … | general(一般) |
| 48 | [45022](https://github.com/vllm-project/vllm/pull/45022) | [Bugfix][Model] Voxtral realtime: fix boot OOM and engine c… | crash(崩溃) |
| 48 | [59098](https://github.com/vllm-project/vllm/pull/59098) | [Bugfix][ROCm] Fix ROCM_AITER_FA context loss on out-of-ord… | general(一般) |
| 46 | [61749](https://github.com/vllm-project/vllm/pull/61749) | [Bugfix][MoE] Require exact strides for moe_forward custom … | general(一般) |

## vllm:Top 作者 (前 15)

| 作者 | 修复 PR 数 |
| --- | --- |
| [njhill](https://github.com/njhill) | 220 |
| [AndreasKaratzas](https://github.com/AndreasKaratzas) | 184 |
| [mgoin](https://github.com/mgoin) | 180 |
| [yewentao256](https://github.com/yewentao256) | 171 |
| [DarkLight1337](https://github.com/DarkLight1337) | 170 |
| [WoosukKwon](https://github.com/WoosukKwon) | 162 |
| [hmellor](https://github.com/hmellor) | 157 |
| [Isotr0py](https://github.com/Isotr0py) | 153 |
| [LucasWilkinson](https://github.com/LucasWilkinson) | 122 |
| [youkaichao](https://github.com/youkaichao) | 98 |
| [jeejeelee](https://github.com/jeejeelee) | 97 |
| [NickLucche](https://github.com/NickLucche) | 76 |
| [chaunceyjiang](https://github.com/chaunceyjiang) | 74 |
| [ywang96](https://github.com/ywang96) | 72 |
| [khluu](https://github.com/khluu) | 70 |

## sglang:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 294 | 53 |
| MoE(混合专家) | 395 | 48 |
| Quantization(量化) | 381 | 50 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 207 | 24 |
| Overlap / Scheduler(重叠与调度) | 229 | 32 |
| Attention / Kernel(注意力与算子) | 667 | 81 |
| KV Cache / Memory(KV Cache 与显存) | 280 | 25 |
| Parallelism / Communication(并行与通信) | 81 | 10 |
| Sampler / Detokenization(采样与反分词) | 68 | 9 |
| Frontend / Serving(前端与服务) | 388 | 27 |
| Model-specific Fast Path(模型专属快路径) | 564 | 48 |
| CI / Test-Perf(CI 与测试性能) | 124 | 0 |
| Other(其他) | 2941 | 210 |

## sglang:bug 类型分布

| 类型 | 总数 | 🔥 重点 |
| --- | --- | --- |
| correctness(正确性) | 307 | 306 |
| crash(崩溃) | 202 | 90 |
| hang(挂死) | 93 | 30 |
| resource(资源泄漏/OOM) | 77 | 17 |
| regression(回归) | 122 | 120 |
| startup(启动/加载) | 90 | 1 |
| general(一般) | 5728 | 53 |

## sglang:年度分布(按合并时间)

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 4184 | 399 |
| 2025 | 2003 | 192 |
| 2024 | 432 | 26 |

## sglang:热度 Top 20 (reactions×2 + comments)

| 热度 | PR | 标题 | 类型 |
| --- | --- | --- | --- |
| 106 | [26203](https://github.com/sgl-project/sglang/pull/26203) | Fix MLA scaling when YARN scaling is disabled | general(一般) |
| 76 | [29792](https://github.com/sgl-project/sglang/pull/29792) | Fix Mamba track-boundary bookkeeping under overlap scheduli… | general(一般) |
| 73 | [23776](https://github.com/sgl-project/sglang/pull/23776) | [DeepSeek V4] Fix meaningless numbers in chat output by add… | general(一般) |
| 60 | [27010](https://github.com/sgl-project/sglang/pull/27010) | [HiCache] Fix PP inconsistency with HiCache L3 (#22607) | general(一般) |
| 52 | [3709](https://github.com/sgl-project/sglang/pull/3709) | Fix allgather ops inside cuda graphs | general(一般) |
| 49 | [18211](https://github.com/sgl-project/sglang/pull/18211) | [CI] Fix torchaudio/torchvision CUDA version mismatch | correctness(正确性) |
| 47 | [2601](https://github.com/sgl-project/sglang/pull/2601) | [Feature, Hardware] Enable DeepseekV3 on AMD GPUs | general(一般) |
| 40 | [20984](https://github.com/sgl-project/sglang/pull/20984) | Fix DeepSeek V32 FP4 test | general(一般) |
| 39 | [33743](https://github.com/sgl-project/sglang/pull/33743) | [Fix] moe: Fix flashinfer TRT-LLM BF16 expert weight reload… | general(一般) |
| 39 | [34668](https://github.com/sgl-project/sglang/pull/34668) | fix(test): stabilize nightly precision regression | correctness(正确性) |
| 37 | [6026](https://github.com/sgl-project/sglang/pull/6026) | [Fix] Reduce busy polling when scheduler is idle | general(一般) |
| 36 | [24954](https://github.com/sgl-project/sglang/pull/24954) | [Mamba] Fix extra_buffer overlap schedule races | general(一般) |
| 35 | [20310](https://github.com/sgl-project/sglang/pull/20310) | [tokenizer] improve non streaming request processing + some… | general(一般) |
| 35 | [32963](https://github.com/sgl-project/sglang/pull/32963) | [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces int… | general(一般) |
| 34 | [33395](https://github.com/sgl-project/sglang/pull/33395) | [Feat] speculative: Seed rejection-sampling draft proposals… | general(一般) |
| 34 | [33922](https://github.com/sgl-project/sglang/pull/33922) | Fix Qwen3.5 GDN multi-item scoring | general(一般) |
| 33 | [33614](https://github.com/sgl-project/sglang/pull/33614) | [Spec] Fix Dspark and Dflash state divergence across TP rank | correctness(正确性) |
| 33 | [34535](https://github.com/sgl-project/sglang/pull/34535) | [Fix] hybrid-SSM: DP attention failures at low concurrency | general(一般) |
| 32 | [29432](https://github.com/sgl-project/sglang/pull/29432) | Fix bookkeeping fields not encapsulated with real allocatio… | general(一般) |
| 32 | [41916](https://github.com/sgl-project/sglang/pull/41916) | [Fix] sampling: Make top-k/top-p filtering consistent acros… | general(一般) |

## sglang:Top 作者 (前 15)

| 作者 | 修复 PR 数 |
| --- | --- |
| [hnyls2002](https://github.com/hnyls2002) | 380 |
| [merrymercy](https://github.com/merrymercy) | 312 |
| [mickqian](https://github.com/mickqian) | 212 |
| [fzyzcjy](https://github.com/fzyzcjy) | 200 |
| [zhyncs](https://github.com/zhyncs) | 177 |
| [ch-wan](https://github.com/ch-wan) | 162 |
| [Fridge003](https://github.com/Fridge003) | 134 |
| [alphabetc1](https://github.com/alphabetc1) | 121 |
| [BBuf](https://github.com/BBuf) | 117 |
| [mmangkad](https://github.com/mmangkad) | 114 |
| [ispobock](https://github.com/ispobock) | 110 |
| [Kangyan-Zhou](https://github.com/Kangyan-Zhou) | 109 |
| [ShangmingCai](https://github.com/ShangmingCai) | 99 |
| [JustinTong0323](https://github.com/JustinTong0323) | 92 |
| [alisonshao](https://github.com/alisonshao) | 92 |

## 数据快照(下次更新的基线)

- 快照记录时间: 2026-11-19T15:41:28+00:00
- **vllm** @8283763f5856 (upstream HEAD 2026-11-19) — commit: https://github.com/vllm-project/vllm/commit/8283763f58569d9e90e998b005d02463de33449c | API 抓取窗口: 2026-11-08T09:35:42+00:00 ~ 2026-11-19T15:37:18+00:00 | raw 8205 → 重点档 663 → 修复卡 655
- **sglang** @5e32149fd6b5 (upstream HEAD 2026-11-19) — commit: https://github.com/sgl-project/sglang/commit/5e32149fd6b58a30a4a72bbaa779aafc1bb6119a | API 抓取窗口: 2026-11-08T10:04:47+00:00 ~ 2026-11-19T15:41:08+00:00 | raw 6619 → 重点档 617 → 修复卡 610
