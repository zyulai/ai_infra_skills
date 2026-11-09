# SGLANG RFC 索引(label:rfc ∪ 标题含 RFC,全量)

- 仓库: https://github.com/sgl-project/sglang
- 收录总数: **202** (重点 🔥 **42** / 一般 160) | 开放 68 / 已关闭 134
- 生成时间: 2026-11-08T02:05:32+00:00

## 目录

- [Speculative Decoding(投机解码) (9)](#speculative-decoding投机解码-9)
- [MoE(混合专家) (6)](#moe混合专家-6)
- [Quantization(量化) (12)](#quantization量化-12)
- [Disaggregated Serving(分离式部署) (12)](#disaggregated-serving分离式部署-12)
- [CUDA Graph / Compiler(CUDA Graph 与编译) (4)](#cuda-graph--compilercuda-graph-与编译-4)
- [Overlap / Scheduler(重叠与调度) (11)](#overlap--scheduler重叠与调度-11)
- [Attention / Kernel(注意力与算子) (13)](#attention--kernel注意力与算子-13)
- [KV Cache / Memory(KV Cache 与显存) (22)](#kv-cache--memorykv-cache-与显存-22)
- [Frontend / Serving / API(前端与服务 API) (13)](#frontend--serving--api前端与服务-api-13)
- [Architecture / Engine(架构与引擎) (15)](#architecture--engine架构与引擎-15)
- [Hardware / Backend(硬件与后端) (8)](#hardware--backend硬件与后端-8)
- [Feature / Ecosystem(功能与生态) (18)](#feature--ecosystem功能与生态-18)
- [CI / Infra / Process(CI 与基础流程) (4)](#ci--infra--processci-与基础流程-4)
- [Other(其他) (55)](#other其他-55)

## Speculative Decoding(投机解码) (9)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#14343](https://github.com/sgl-project/sglang/issues/14343) | [Feature] RFC: Data Parallel Draft Model(DPDraft) | DarkraiHL | 2025-12-03 | ⚪ 已关闭 | 一般 |  |  |
| [#15319](https://github.com/sgl-project/sglang/issues/15319) | [Feature] RFC: AutoSpec, Automatic Runtime Speculative Inference Parameter Tuning | maodoudou168 | 2025-12-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-15319-feature-rfc-autospec-automatic-runtime-s.md) |
| [#17767](https://github.com/sgl-project/sglang/issues/17767) | [Feature] [RFC] Extend EAGLE worker to support a step-disaggregated draft model, PRISM | Akemiiii | 2026-01-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-17767-feature-rfc-extend-eagle-worker-to-suppo.md) |
| [#22044](https://github.com/sgl-project/sglang/issues/22044) | [Feature][RFC] SPECTRE: Parallel Speculative Decoding with a Multi-Tenant Remote Drafter | zhangfeiyu5610 | 2026-04-03 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22044-feature-rfc-spectre-parallel-speculative.md) |
| [#28511](https://github.com/sgl-project/sglang/issues/28511) | [RFC] Porting ReplaySSM to SGLang: faster decode and speculative decoding for hybrid (GDN… | yuan-luo | 2026-06-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-28511-rfc-porting-replayssm-to-sglang-faster-d.md) |
| [#30263](https://github.com/sgl-project/sglang/issues/30263) | [Feature] Per-request opt-out of speculative decoding (RFC: disable_speculative_decoding … | anencore94 | 2026-07-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-30263-feature-per-request-opt-out-of-speculati.md) |
| [#45695](https://github.com/sgl-project/sglang/issues/45695) | [RFC]: add eagle4 spec decode | HF-001 | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#46520](https://github.com/sgl-project/sglang/issues/46520) | [RFC] [XPU] paged-experts: Intel XPU support with Lookahead Expert Scheduling (LES) and a… | wyc55069407 | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#47797](https://github.com/sgl-project/sglang/issues/47797) | [RFC] Define per-request speculative decoding opt-out execution semantics | anencore94 | 2026-11-01 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47797-rfc-define-per-request-speculative-decod.md) |

## MoE(混合专家) (6)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#20332](https://github.com/sgl-project/sglang/issues/20332) | [RFC] Faster loading for MoE models using TP | nvjullin | 2026-03-11 | ⚪ 已关闭 | 一般 |  |  |
| [#28663](https://github.com/sgl-project/sglang/issues/28663) | Deprecate and delete `--moe-runner-backend cutlass`, dense FP4 GEMM, FP4 quantization (in… | b8zhong | 2026-06-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-28663-deprecate-and-delete-moe-runner-backend.md) |
| [#32712](https://github.com/sgl-project/sglang/issues/32712) | [Feature] [RFC] SharedEP: a zerocopy execution model for MoE Expert Parallelism | foraxe | 2026-07-29 | ⚪ 已关闭 | 一般 |  |  |
| [#39192](https://github.com/sgl-project/sglang/issues/39192) | [RFC] Contention-aware batching for dynamic EPLB expert migration | DOCCA0 | 2026-09-12 | 🟢 开放 | 一般 |  |  |
| [#43820](https://github.com/sgl-project/sglang/issues/43820) | [RFC] Prefill sequence parallelism with replicated MoE for small-TP long-prompt prefill (… | AliceChenyy | 2026-10-12 | 🟢 开放 | 一般 |  |  |
| [#44697](https://github.com/sgl-project/sglang/issues/44697) | [RFC] MoE routing simulator for prefill imbalance profiling: balanced reference and GPU-f… | Phoenix3334 | 2026-10-16 | 🟢 开放 | 一般 |  |  |

## Quantization(量化) (12)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#15725](https://github.com/sgl-project/sglang/issues/15725) | [RFC] Add Intel Quantization Support in SGLang | mengniwang95 | 2025-12-24 | ⚪ 已关闭 | 一般 |  |  |
| [#20959](https://github.com/sgl-project/sglang/issues/20959) | [RFC] NVIDIA ModelOpt Quantized Diffusion & Video Model Deployment in SGLang | Edwardf0t1 | 2026-03-20 | ⚪ 已关闭 | 一般 |  |  |
| [#21159](https://github.com/sgl-project/sglang/issues/21159) | [RFC][Diffusion] Intel Auto-Round x SGL Diffusion Quantization Support (2026 H1) | WeiweiZhang1 | 2026-03-23 | ⚪ 已关闭 | 一般 |  |  |
| [#21584](https://github.com/sgl-project/sglang/issues/21584) | [RFC][NPU] Ascend NPU A5 Support for MXFP8/MXFP4 Quantization | YChange01 | 2026-03-28 | ⚪ 已关闭 | 一般 |  |  |
| [#21769](https://github.com/sgl-project/sglang/issues/21769) | [RFC] AMD Quark Quantized Diffusion Image&Video Model Deployment in SGLang | ColinZ22 | 2026-03-31 | ⚪ 已关闭 | 一般 |  |  |
| [#28543](https://github.com/sgl-project/sglang/issues/28543) | Deprecate Marlin FP8 quantization, QServe, FBGEMM FP8 | b8zhong | 2026-06-17 | ⚪ 已关闭 | 一般 |  |  |
| [#28618](https://github.com/sgl-project/sglang/issues/28618) | [RFC] Add SM89/L20 support for DeepSeek-V4-Flash-FP8 | xesdiny | 2026-06-18 | ⚪ 已关闭 | 一般 |  |  |
| [#32112](https://github.com/sgl-project/sglang/issues/32112) | Deprecate legacy (non-Marlin) GPTQ kernel and Dual Chunk Flash Attention backend | b8zhong | 2026-07-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32112-deprecate-legacy-non-marlin-gptq-kernel.md) |
| [#36511](https://github.com/sgl-project/sglang/issues/36511) | [Feature/RFC] Optional fp8 (quantized) input embedding to free VRAM for large-vocab model… | angelbaba01L | 2026-08-26 | ⚪ 已关闭 | 一般 |  |  |
| [#38004](https://github.com/sgl-project/sglang/issues/38004) | [RFC] Quantization module layout: Config, Scheme, backend kernels | mmangkad | 2026-09-04 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-38004-rfc-quantization-module-layout-config-sc.md) |
| [#41853](https://github.com/sgl-project/sglang/issues/41853) | [RFC] Gluon MegaMoE: EP8 MXFP4 and EP16 FP8 Support | Sulfur6 | 2026-09-30 | 🟢 开放 | 一般 |  |  |
| [#41866](https://github.com/sgl-project/sglang/issues/41866) | [RFC][NPU] INT8 (C8) KV cache for DSA models (GLM-5.2) on Ascend A2/A3 (910B/910C) | qidaye | 2026-09-30 | 🟢 开放 | 一般 |  |  |

## Disaggregated Serving(分离式部署) (12)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#6925](https://github.com/sgl-project/sglang/issues/6925) | [RFC][Feature] Support Remote Prefill in PD Disaggregation | rainj-me | 2025-06-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6925-rfc-feature-support-remote-prefill-in-pd.md) |
| [#9347](https://github.com/sgl-project/sglang/issues/9347) | [RFC] Support FFN-Attention Disaggregation Strategy for scientific research and low capab… | yiakwy-xpu-ml-framework-team | 2025-08-19 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-9347-rfc-support-ffn-attention-disaggregation.md) |
| [#19512](https://github.com/sgl-project/sglang/issues/19512) | [RFC] [Diffusion] SGLang Diffusion Disaggregation | FredHuang99 | 2026-02-27 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-19512-rfc-diffusion-sglang-diffusion-disaggreg.md) |
| [#23162](https://github.com/sgl-project/sglang/issues/23162) | [RFC] GLM-5 Spec. Decode + PP in Disaggregation Decode Mode | MMuzzammil1 | 2026-04-19 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23162-rfc-glm-5-spec-decode-pp-in-disaggregati.md) |
| [#27622](https://github.com/sgl-project/sglang/issues/27622) | [RFC] Host-Backed Deferred Device KV Allocation for PD Disaggregation | aooxin | 2026-06-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-27622-rfc-host-backed-deferred-device-kv-alloc.md) |
| [#28669](https://github.com/sgl-project/sglang/issues/28669) | [RFC] Disaggregated request preprocessing | sagearc | 2026-06-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-28669-rfc-disaggregated-request-preprocessing.md) |
| [#32841](https://github.com/sgl-project/sglang/issues/32841) | [RFC] Asynchronous completion handling for the HiCache NIXL storage backend | aaltinsoy | 2026-07-30 | ⚪ 已关闭 | 一般 | 4x latency to small O_DIRECT transfers and caps disk-true b… |  |
| [#33861](https://github.com/sgl-project/sglang/issues/33861) | [RFC] PD disaggregation: single protocol layer, per-backend transport | hnyls2002 | 2026-08-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33861-rfc-pd-disaggregation-single-protocol-la.md) |
| [#39740](https://github.com/sgl-project/sglang/issues/39740) | [RFC] Pluggable KV Compression for Disaggregated Serving | luoroger37 | 2026-09-16 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-39740-rfc-pluggable-kv-compression-for-disaggr.md) |
| [#45527](https://github.com/sgl-project/sglang/issues/45527) | [RFC] PD disaggregation: send the last KV chunk from later PP stages as soon as their for… | eaglewangao-us | 2026-10-21 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45527-rfc-pd-disaggregation-send-the-last-kv-c.md) |
| [#45568](https://github.com/sgl-project/sglang/issues/45568) | [RFC] Attention-FFN disaggregation: design and integration plan | sunfanyi | 2026-10-21 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45568-rfc-attention-ffn-disaggregation-design.md) |
| [#47031](https://github.com/sgl-project/sglang/issues/47031) | [RFC] Phased upstreaming of reliable Mooncake Direct L3 loading and promotion-based prefe… | lihenga | 2026-10-28 | 🟢 开放 | 一般 |  |  |

## CUDA Graph / Compiler(CUDA Graph 与编译) (4)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#10118](https://github.com/sgl-project/sglang/issues/10118) | [RFC] SGLang unified kernel fusion and torch compile optimisations | DevashishLal-CB | 2025-09-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-10118-rfc-sglang-unified-kernel-fusion-and-tor.md) |
| [#23004](https://github.com/sgl-project/sglang/issues/23004) | [RFC] Cuda Graph Runner Backend Refactor | Oasis-Git | 2026-04-16 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23004-rfc-cuda-graph-runner-backend-refactor.md) |
| [#32432](https://github.com/sgl-project/sglang/issues/32432) | [RFC] Define Metadata, Workspace, and Stream-Ownership Contracts for Dynamic CUDA Graph R… | Phoenix3334 | 2026-07-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32432-rfc-define-metadata-workspace-and-stream.md) |
| [#33852](https://github.com/sgl-project/sglang/issues/33852) | [RFC] Allow prefill CUDA graph bs below chunked_prefill_size under post-capture KV sizing | hnyls2002 | 2026-08-06 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33852-rfc-allow-prefill-cuda-graph-bs-below-ch.md) |

## Overlap / Scheduler(重叠与调度) (11)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#16787](https://github.com/sgl-project/sglang/issues/16787) | [Feature] RFC: Shared-Queue Fast Path for Tokenizer→Scheduler Request Handoff (Multimodal) | Zhangpch2021 | 2026-01-09 | ⚪ 已关闭 | 一般 |  |  |
| [#27898](https://github.com/sgl-project/sglang/issues/27898) | [RFC] Beyond Passive Byte Stores: Scheduler-Aware Multi-Tier KV Caching for SGLang with M… | TianDi101 | 2026-06-11 | ⚪ 已关闭 | 一般 |  |  |
| [#28186](https://github.com/sgl-project/sglang/issues/28186) | [RFC] A Policy-Programmable Scheduling Interface for SGLang Diffusion | tie-pilot-qxw | 2026-06-14 | ⚪ 已关闭 | 一般 |  |  |
| [#29573](https://github.com/sgl-project/sglang/issues/29573) | [Feature] RFC: Paper Reproduction of LMetric Multiplication Scheduling in SGLang Gateway | sjmshsh | 2026-06-28 | ⚪ 已关闭 | 一般 |  |  |
| [#32271](https://github.com/sgl-project/sglang/issues/32271) | [RFC][Feature] Topology-transparent multi-NIC HTTP ingress with a single logical scheduli… | catyans | 2026-07-24 | ⚪ 已关闭 | 一般 |  |  |
| [#43411](https://github.com/sgl-project/sglang/issues/43411) | [RFC] Native worker batching for ComfyUI diffusion requests | OnePunchMonk | 2026-10-09 | 🟢 开放 | 一般 |  |  |
| [#44721](https://github.com/sgl-project/sglang/issues/44721) | [RFC] scheduler: Overlap prefill scheduling and KV loading with model execution | ttaohe | 2026-10-16 | 🟢 开放 | 一般 |  |  |
| [#45515](https://github.com/sgl-project/sglang/issues/45515) | [RFC][DSpark] Online latency correction for confidence-scheduled verification | SuperGoodGame | 2026-10-21 | 🟢 开放 | 一般 |  |  |
| [#47019](https://github.com/sgl-project/sglang/issues/47019) | [RFC] Decouple Scheduler Control from Forward Submission | froststeam | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#47289](https://github.com/sgl-project/sglang/issues/47289) | [RFC] Add per-step TBO admission policy and fallback diagnostics | algebra-MCX | 2026-10-29 | ⚪ 已关闭 | 一般 |  |  |
| [#48519](https://github.com/sgl-project/sglang/issues/48519) | [RFC] scheduler: UniBoost, a starvation-bounded schedule policy (#48076-#48079) | phphlio1 | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## Attention / Kernel(注意力与算子) (13)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#13632](https://github.com/sgl-project/sglang/issues/13632) | [RFC] Proposal: Split sgl-kernel and sgl-router into separate repositories | WingEdge777 | 2025-11-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-13632-rfc-proposal-split-sgl-kernel-and-sgl-ro.md) |
| [#17110](https://github.com/sgl-project/sglang/issues/17110) | [RFC] sgl-kernel Release Strategy Under Current Storage Quota Constraints | zhyncs | 2026-01-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-17110-rfc-sgl-kernel-release-strategy-under-cu.md) |
| [#18237](https://github.com/sgl-project/sglang/issues/18237) | [RFC] Complexity in model layers and communication kernels | nvjullin | 2026-02-04 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18237-rfc-complexity-in-model-layers-and-commu.md) |
| [#18365](https://github.com/sgl-project/sglang/issues/18365) | 🚀 [RFC] Decoupling Framework Logic from Hardware Kernels: Offloading Triton Kernels to sg… | pre-master | 2026-02-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18365-rfc-decoupling-framework-logic-from-hard.md) |
| [#23058](https://github.com/sgl-project/sglang/issues/23058) | RFC: optional HIP external LLMM1 fast path for M=1 decode GEMMs | skyguan92 | 2026-04-17 | ⚪ 已关闭 | 一般 |  |  |
| [#27652](https://github.com/sgl-project/sglang/issues/27652) | [RFC] ParisKV2 sparse attention integration for SGLang | amy-77 | 2026-06-09 | ⚪ 已关闭 | 一般 |  |  |
| [#28490](https://github.com/sgl-project/sglang/issues/28490) | [RFC][NPU][DLLM] Add fused tracker and coreset kernels for LLaDA2 JointThreshold | leedongkun30-arch | 2026-06-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-28490-rfc-npu-dllm-add-fused-tracker-and-cores.md) |
| [#29630](https://github.com/sgl-project/sglang/issues/29630) | [RFC] Introduce a unified sglang.kernels namespace for kernel organization and dispatch | BBuf | 2026-06-29 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-29630-rfc-introduce-a-unified-sglang-kernels-n.md) |
| [#32111](https://github.com/sgl-project/sglang/issues/32111) | Deprecate CUTLASS MLA attention backend | b8zhong | 2026-07-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32111-deprecate-cutlass-mla-attention-backend.md) |
| [#32657](https://github.com/sgl-project/sglang/issues/32657) | [RFC] A Unified KV-Cache Sparsity Framework for Post-Hoc Sparse Attention | magicYang1573 | 2026-07-28 | 🟢 开放 | 一般 |  |  |
| [#40522](https://github.com/sgl-project/sglang/issues/40522) | [RFC] Decouple the SWA sidecar page size from the full-attention page size (DSv4, Hybrid … | mitu626 | 2026-09-21 | 🟢 开放 | 一般 |  |  |
| [#43949](https://github.com/sgl-project/sglang/issues/43949) | [RFC] Monokernel framework for SGLang | kevin-mii | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#44786](https://github.com/sgl-project/sglang/issues/44786) | [RFC / Feature] SPICY-KV: 28.6x–50x Pre-RoPE Algebraic Multigrid KV Compression for Radix… | Yuge-Y | 2026-10-16 | 🟢 开放 | 一般 |  |  |

## KV Cache / Memory(KV Cache 与显存) (22)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#7746](https://github.com/sgl-project/sglang/issues/7746) | [RFC] Remote KV Connector for SGLang Global Cache Reuse and PD | hzh0425 | 2025-07-03 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7746-rfc-remote-kv-connector-for-sglang-globa.md) |
| [#21542](https://github.com/sgl-project/sglang/issues/21542) | [Feature] [RFC] HiCache retention slots | lawrence-harmonic | 2026-03-27 | ⚪ 已关闭 | 一般 |  |  |
| [#24542](https://github.com/sgl-project/sglang/issues/24542) | [RFC] [HiCache] A new backend for external L3 storage systems | stepinto | 2026-05-06 | ⚪ 已关闭 | 一般 |  |  |
| [#24656](https://github.com/sgl-project/sglang/issues/24656) | [RFC] Agent-Aware KV Cache Phase 1 for Agentic Workloads | stmatengss | 2026-05-08 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-24656-rfc-agent-aware-kv-cache-phase-1-for-age.md) |
| [#27310](https://github.com/sgl-project/sglang/issues/27310) | [RFC] GPU Memory Service (GMS) integration for out-of-process GPU memory management in SG… | galletas1712 | 2026-06-04 | ⚪ 已关闭 | 一般 |  |  |
| [#28215](https://github.com/sgl-project/sglang/issues/28215) | [RFC] Optimize RadixKey.match with native prefix comparison | merrymercy | 2026-06-14 | ⚪ 已关闭 | 一般 |  |  |
| [#28420](https://github.com/sgl-project/sglang/issues/28420) | [RFC] Migrate UnifiedRadixCache logical backbone to Rust | ispobock | 2026-06-16 | ⚪ 已关闭 | 一般 |  |  |
| [#29709](https://github.com/sgl-project/sglang/issues/29709) | [RFC]: KV Cache Events for HiCache L3 Storage Backends | stmatengss | 2026-06-30 | 🟢 开放 | 一般 |  |  |
| [#30928](https://github.com/sgl-project/sglang/issues/30928) | [RFC] Position-Independent KV Cache Reuse for Agentic/RAG Workloads | krakhit | 2026-07-12 | ⚪ 已关闭 | 🔥 重点 | Claims 51-94% TTFT reduction, 3x throughput, tested on Deep… | [卡片](rfc-cards/rfc-30928-rfc-position-independent-kv-cache-reuse.md) |
| [#31458](https://github.com/sgl-project/sglang/issues/31458) | [Feature] RFC: SGLang KV Indexer for Distributed KV Cache Placement Metadata | wuyl1 | 2026-07-16 | ⚪ 已关闭 | 一般 |  |  |
| [#31779](https://github.com/sgl-project/sglang/issues/31779) | [RFC] An NPU-Native, Sparsity-Driven KV Offloading SubSystem for Efficient LLM Decoding | nzbej | 2026-07-20 | ⚪ 已关闭 | 一般 |  |  |
| [#32903](https://github.com/sgl-project/sglang/issues/32903) | [RFC] KVCR as a HiCacheStorage backend for peer-to-peer KV reuse | linhu-nv | 2026-07-30 | ⚪ 已关闭 | 一般 |  |  |
| [#35620](https://github.com/sgl-project/sglang/issues/35620) | [RFC] Integrating Ascend MemCache as an HiCache L3 Backend | gjsheu | 2026-08-20 | ⚪ 已关闭 | 一般 |  |  |
| [#36522](https://github.com/sgl-project/sglang/issues/36522) | [RFC] KV Cache Compression API (KVTC, nvCOMP, ...) | alancucki | 2026-08-26 | 🟢 开放 | 一般 |  |  |
| [#37372](https://github.com/sgl-project/sglang/issues/37372) | [Feature] [RFC] [HiCache] Out-of-process HiCache data plane with device-memory IPC | imReese | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#39991](https://github.com/sgl-project/sglang/issues/39991) | [RFC] Align KV cache events with vLLM's schema so shared consumers have a standard format | hickeyma | 2026-09-17 | 🟢 开放 | 一般 |  |  |
| [#40865](https://github.com/sgl-project/sglang/issues/40865) | [RFC] Explicit, budgeted tail-replay for Mamba/GDN state in UnifiedRadixCache | curnane-lab | 2026-09-23 | ⚪ 已关闭 | 一般 |  |  |
| [#41514](https://github.com/sgl-project/sglang/issues/41514) | [RFC / HiCache] Same-node peer L2 sharing across DP ranks via /dev/shm | VNG-VMONITOR | 2026-09-28 | 🟢 开放 | 一般 |  |  |
| [#44171](https://github.com/sgl-project/sglang/issues/44171) | [RFC] Preserve multimodal identity end to end: precomputed inputs, collision-safe KV reus… | michaelfeil | 2026-10-14 | 🟢 开放 | 一般 |  |  |
| [#45189](https://github.com/sgl-project/sglang/issues/45189) | [RFC] Session-Aware KV Cache Orchestration for SGLang Router and HiCache | zhhangBian | 2026-10-19 | 🟢 开放 | 一般 |  |  |
| [#47021](https://github.com/sgl-project/sglang/issues/47021) | [RFC]: Quota and global-eviction contract for HiCache L3 when the backend is a shared fil… | kdtmac | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#48144](https://github.com/sgl-project/sglang/issues/48144) | [RFC] KV Cache Optimization for Cosmos3-Nano-Sim-Bimanual | xupinjie | 2026-11-03 | 🟢 开放 | 一般 |  |  |

## Frontend / Serving / API(前端与服务 API) (13)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#7453](https://github.com/sgl-project/sglang/issues/7453) | [RFC] Move request disconnection handling out of tokenizer manager | CatherineSue | 2025-06-23 | ⚪ 已关闭 | 一般 |  |  |
| [#11141](https://github.com/sgl-project/sglang/issues/11141) | [RFC] [Feature] Improving request-level timing and performance metrics | scottjlee | 2025-10-01 | ⚪ 已关闭 | 一般 |  |  |
| [#13631](https://github.com/sgl-project/sglang/issues/13631) | [RFC] Router disconnect/abort requests from PD pairs when detect one of it down. | llc-kc | 2025-11-20 | ⚪ 已关闭 | 一般 |  |  |
| [#15448](https://github.com/sgl-project/sglang/issues/15448) | [Feature] RFC: Introducing Native Async Support for SGLang Frontend Language | Peng-YM | 2025-12-19 | ⚪ 已关闭 | 一般 |  |  |
| [#22558](https://github.com/sgl-project/sglang/issues/22558) | [RFC] Native gRPC Server for SGLang in Rust | alexnails | 2026-04-10 | ⚪ 已关闭 | 一般 |  |  |
| [#24467](https://github.com/sgl-project/sglang/issues/24467) | [Feature][RFC] Plug-in metrics backend via class-level DI for collectors | nadongjun | 2026-05-06 | ⚪ 已关闭 | 一般 |  |  |
| [#26343](https://github.com/sgl-project/sglang/issues/26343) | [RFC] Streaming API for Real-Time Server State and Metrics | liu-cong | 2026-05-26 | ⚪ 已关闭 | 一般 |  |  |
| [#32925](https://github.com/sgl-project/sglang/issues/32925) | [RFC] Push-based Engine Load Reporting and Router Load Monitoring | Bakerjc-bgner | 2026-07-30 | ⚪ 已关闭 | 一般 |  |  |
| [#33394](https://github.com/sgl-project/sglang/issues/33394) | [RFC][sgl-router] Recoverable KV Placement State with Snapshots and Event Replay | skajre | 2026-08-03 | ⚪ 已关闭 | 一般 |  |  |
| [#34513](https://github.com/sgl-project/sglang/issues/34513) | [RFC] Agent-aware session affinity without routing keys: a router policy | Biaogezi | 2026-08-12 | ⚪ 已关闭 | 一般 |  |  |
| [#37389](https://github.com/sgl-project/sglang/issues/37389) | [RFC][Feature] Extract an In-Process Route Planner from experimental/sgl-router's PD Work… | tonyluj | 2026-09-01 | ⚪ 已关闭 | 一般 |  |  |
| [#44374](https://github.com/sgl-project/sglang/issues/44374) | [Feature] [RFC] Opt-in `toktier` tokenizer backend: exact incremental and chunk-parallel … | ae86zhizhi | 2026-10-15 | 🟢 开放 | 一般 |  |  |
| [#48232](https://github.com/sgl-project/sglang/issues/48232) | [RFC] Engine-free render service for sglang-processor over gRPC (Lower/Respond), includin… | ahg-g | 2026-11-03 | 🟢 开放 | 一般 |  |  |

## Architecture / Engine(架构与引擎) (15)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#9853](https://github.com/sgl-project/sglang/issues/9853) | RFC: Command-Line Interface Architecture for SGLang | vincentzed | 2025-09-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-9853-rfc-command-line-interface-architecture.md) |
| [#10464](https://github.com/sgl-project/sglang/issues/10464) | [Feature] RFC: Integrating checkpoint engine into SGLang | stmatengss | 2025-09-15 | ⚪ 已关闭 | 一般 |  |  |
| [#19139](https://github.com/sgl-project/sglang/issues/19139) | [Rfc] Refactor server_args.py | vincentzed | 2026-02-22 | ⚪ 已关闭 | 一般 |  |  |
| [#20481](https://github.com/sgl-project/sglang/issues/20481) | [RFC] Refactor server_args.py | Fridge003 | 2026-03-13 | ⚪ 已关闭 | 一般 |  |  |
| [#24703](https://github.com/sgl-project/sglang/issues/24703) | [RFC] Refactor weight loading code | b8zhong | 2026-05-08 | 🟢 开放 | 一般 |  |  |
| [#24945](https://github.com/sgl-project/sglang/issues/24945) | [RFC] SGLang EPD Performance / Architecture / Observability Enhancements | Abatom | 2026-05-11 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-24945-rfc-sglang-epd-performance-architecture.md) |
| [#25371](https://github.com/sgl-project/sglang/issues/25371) | [RFC][Refactor] `mem_cache` pool / allocator restructure | alphabetc1 | 2026-05-15 | ⚪ 已关闭 | 一般 |  |  |
| [#27052](https://github.com/sgl-project/sglang/issues/27052) | [Feature] [RFC] Fast Recovery for sglang engine | QiuMike | 2026-06-02 | ⚪ 已关闭 | 一般 |  |  |
| [#29363](https://github.com/sgl-project/sglang/issues/29363) | [RFC] Cross-architecture model reload: `reload_model()` for architecture-changing model s… | AlanFokCo | 2026-06-26 | ⚪ 已关闭 | 一般 |  |  |
| [#35376](https://github.com/sgl-project/sglang/issues/35376) | [RFC] Orchestrator-facing Elastic EP lifecycle contract | julienmancuso | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#36431](https://github.com/sgl-project/sglang/issues/36431) | [RFC] Resumable raw token stream for /v1/chat/completions | RagingSilence | 2026-08-26 | ⚪ 已关闭 | 一般 |  |  |
| [#43965](https://github.com/sgl-project/sglang/issues/43965) | [Feature] [RFC] SGLang Engine Initialization Snapshots: Reusing Initialized State for Fas… | fuhefei | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#45416](https://github.com/sgl-project/sglang/issues/45416) | [RFC] pd: Prefill short requests on the decode worker that caches their prefix (PPD) | alphabetc1 | 2026-10-20 | 🟢 开放 | 一般 |  |  |
| [#45816](https://github.com/sgl-project/sglang/issues/45816) | [RFC] Complete migration to the shared api.v1 backend API | sagearc | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#48822](https://github.com/sgl-project/sglang/issues/48822) | [RFC] Add NPU RLHF Lifecycle Tests | floatlibai | 2026-11-06 | 🟢 开放 | 一般 |  |  |

## Hardware / Backend(硬件与后端) (8)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#2807](https://github.com/sgl-project/sglang/issues/2807) | [Feature] RFC for adding CPU support for SGLang | mingfeima | 2025-01-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-2807-feature-rfc-for-adding-cpu-support-for-s.md) |
| [#20224](https://github.com/sgl-project/sglang/issues/20224) | [RFC] Introduce PluggableLayer Decorator-Based Restration for Out-of-Tree Layer Raplaceme… | wm901115nwpu | 2026-03-10 | ⚪ 已关闭 | 一般 |  |  |
| [#26426](https://github.com/sgl-project/sglang/issues/26426) | [RFC] Building towards a Hardware Abstraction Layer in SGLang | DevashishLal-CB | 2026-05-27 | ⚪ 已关闭 | 一般 |  |  |
| [#34562](https://github.com/sgl-project/sglang/issues/34562) | [RFC] HIP/ROCm - Provide a bit-exact SWA prefix reuse for DeepSeek-V4 with unified_kv att… | amd-danli103 | 2026-08-12 | ⚪ 已关闭 | 一般 |  |  |
| [#35709](https://github.com/sgl-project/sglang/issues/35709) | [RFC][NPU] Ascend Cross-Generation Device Adaptation | wangyao-i | 2026-08-20 | ⚪ 已关闭 | 一般 |  |  |
| [#39521](https://github.com/sgl-project/sglang/issues/39521) | [RFC] FuseEP Integration for MiniMax-M3 Prefill on Ascend NPU | ClownBin | 2026-09-15 | 🟢 开放 | 一般 |  |  |
| [#45547](https://github.com/sgl-project/sglang/issues/45547) | [RFC] adapter hisparse to npu | HF-001 | 2026-10-21 | 🟢 开放 | 一般 |  |  |
| [#47624](https://github.com/sgl-project/sglang/issues/47624) | [RFC] NPU RL Test Consolidation: File Ownership, Coverage Docstrings, and CI Integration … | SeaAsters | 2026-10-31 | 🟢 开放 | 一般 |  |  |

## Feature / Ecosystem(功能与生态) (18)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5471](https://github.com/sgl-project/sglang/issues/5471) | [RFC][Feature][Model] Add templated fallback HF `transformers` model backend in SRT | XuehaiPan | 2025-04-16 | ⚪ 已关闭 | 一般 |  |  |
| [#7009](https://github.com/sgl-project/sglang/issues/7009) | [RFC] Support Multi-Stage Awake for RL | hebiao064 | 2025-06-09 | ⚪ 已关闭 | 一般 |  |  |
| [#7941](https://github.com/sgl-project/sglang/issues/7941) | [RFC] Multi-framework support for sglang | wangtiance | 2025-07-11 | ⚪ 已关闭 | 一般 |  |  |
| [#12910](https://github.com/sgl-project/sglang/issues/12910) | [Feature] RFC: Add a planner to manage seed instances for loading weights from remote ins… | tianyuzhou95 | 2025-11-09 | ⚪ 已关闭 | 一般 |  |  |
| [#15299](https://github.com/sgl-project/sglang/issues/15299) | [RFC] Unifying Multi-Vendor Support via a Platform Interface in sglang | yeahdongcn | 2025-12-17 | ⚪ 已关闭 | 一般 |  |  |
| [#17669](https://github.com/sgl-project/sglang/issues/17669) | [RFC] R-Fork support nnodes and DP | amysaq2023 | 2026-01-24 | ⚪ 已关闭 | 一般 |  |  |
| [#18162](https://github.com/sgl-project/sglang/issues/18162) | [Feature] RFC: Add msProbe dump support in SGLang | is-not | 2026-02-03 | ⚪ 已关闭 | 一般 |  |  |
| [#19758](https://github.com/sgl-project/sglang/issues/19758) | [RFC] Add InstantTensor Support in SGLang | arlo-scitix | 2026-03-03 | ⚪ 已关闭 | 一般 |  |  |
| [#24653](https://github.com/sgl-project/sglang/issues/24653) | [Feature][RFC] Add `SGLANG_RAY_BUNDLE_INDICES` for fine-grained Ray bundle index control | KaisennHu | 2026-05-08 | ⚪ 已关闭 | 一般 |  |  |
| [#26438](https://github.com/sgl-project/sglang/issues/26438) | [RFC] In-Tree Support for Cambricon MLU Backend | chenxb002 | 2026-05-27 | ⚪ 已关闭 | 一般 |  |  |
| [#26702](https://github.com/sgl-project/sglang/issues/26702) | [RFC] TensorCast KV backend integration | zhou-yuhan | 2026-05-29 | ⚪ 已关闭 | 一般 |  |  |
| [#29119](https://github.com/sgl-project/sglang/issues/29119) | [RFC] Add Top-nσ (Top-n-sigma) logit truncation as a built-in CustomLogitProcessor | Jackie2049 | 2026-06-24 | ⚪ 已关闭 | 一般 |  |  |
| [#32300](https://github.com/sgl-project/sglang/issues/32300) | [RFC] Add CI Infrastructure for the SGLang MLU Backend | chenxb002 | 2026-07-24 | 🟢 开放 | 一般 |  |  |
| [#34295](https://github.com/sgl-project/sglang/issues/34295) | [RFC] Remove the torchao integration (`--torchao-config`) | b8zhong | 2026-08-10 | ⚪ 已关闭 | 一般 |  |  |
| [#38334](https://github.com/sgl-project/sglang/issues/38334) | [RFC] Gluon MegaMoE: SGLang Integration and Multi-Node Support | Sulfur6 | 2026-09-07 | 🟢 开放 | 一般 |  |  |
| [#44221](https://github.com/sgl-project/sglang/issues/44221) | [RFC][Simulator] Support Closed-Loop Agentic Workloads | Jeff010101 | 2026-10-14 | 🟢 开放 | 一般 |  |  |
| [#44635](https://github.com/sgl-project/sglang/issues/44635) | [RFC] Context-parallel support for the unified L3 key scheme | momaek | 2026-10-16 | 🟢 开放 | 一般 |  |  |
| [#48582](https://github.com/sgl-project/sglang/issues/48582) | [RFC] Integrate MindIE-SD as an Acceleration Library for SGLang Diffusion | lynxxxi | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## CI / Infra / Process(CI 与基础流程) (4)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#9808](https://github.com/sgl-project/sglang/issues/9808) | RFC: SGLang Benchmarking Script Consolidation and Reliability Improvements | vincentzed | 2025-08-30 | ⚪ 已关闭 | 一般 |  |  |
| [#16843](https://github.com/sgl-project/sglang/issues/16843) | [Feature] RFC: Add Ag-LiveCodeBench-X language-agnostic benchmark | blazingbhavneek | 2026-01-10 | ⚪ 已关闭 | 一般 |  |  |
| [#32321](https://github.com/sgl-project/sglang/issues/32321) | [RFC] Apple Silicon serving redesign: Torch-owned SRT path with an exported whole-model M… | noob-se7en | 2026-07-24 | 🟢 开放 | 一般 |  |  |
| [#46764](https://github.com/sgl-project/sglang/issues/46764) | [RFC][Feature] SM90 Implementation Plan for the Q4 Agentic Cache Scheduling Roadmap | luoroger37 | 2026-10-27 | ⚪ 已关闭 | 一般 |  |  |

## Other(其他) (55)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5931](https://github.com/sgl-project/sglang/issues/5931) | [Feature][RFC] A `sglang serve` subcommand | bzhong-solink | 2025-04-30 | ⚪ 已关闭 | 一般 |  |  |
| [#6006](https://github.com/sgl-project/sglang/issues/6006) | [RFC] sm75 EOL | zhyncs | 2025-05-04 | ⚪ 已关闭 | 一般 |  |  |
| [#7332](https://github.com/sgl-project/sglang/issues/7332) | [RFC] Bi-weekly release | zhyncs | 2025-06-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7332-rfc-bi-weekly-release.md) |
| [#7427](https://github.com/sgl-project/sglang/issues/7427) | 🚧  RFC: Redesign Batch Processing as an Offline Workflow | CatherineSue | 2025-06-21 | ⚪ 已关闭 | 一般 |  |  |
| [#7851](https://github.com/sgl-project/sglang/issues/7851) | [RFC] DRI for every module | zhyncs | 2025-07-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7851-rfc-dri-for-every-module.md) |
| [#9781](https://github.com/sgl-project/sglang/issues/9781) | [RFC] Consolidate SGLang environment variables | vincentzed | 2025-08-29 | ⚪ 已关闭 | 一般 |  |  |
| [#11334](https://github.com/sgl-project/sglang/issues/11334) | [RFC] Directory Structure Does Not Align with Non-PyTorch Implementations | chz34 | 2025-10-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-11334-rfc-directory-structure-does-not-align-w.md) |
| [#12766](https://github.com/sgl-project/sglang/issues/12766) | [Feature] RFC: Block Diffusion Large Language Model (dLLM) Framework In SGLang | ClawSeven | 2025-11-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-12766-feature-rfc-block-diffusion-large-langua.md) |
| [#15070](https://github.com/sgl-project/sglang/issues/15070) | [Feature][Gateway][RFC] SLA-aware PD Routing to Meet TTFT/TPOT and Increase Throughput | qianlihuang | 2025-12-13 | ⚪ 已关闭 | 一般 |  |  |
| [#15502](https://github.com/sgl-project/sglang/issues/15502) | [RFC] SageMaker Day-0 Compatability with SGLang | zhaozuy | 2025-12-19 | ⚪ 已关闭 | 一般 |  |  |
| [#16546](https://github.com/sgl-project/sglang/issues/16546) | [RFC] SGLang-Omni Design | FrankLeeeee | 2026-01-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-16546-rfc-sglang-omni-design.md) |
| [#19421](https://github.com/sgl-project/sglang/issues/19421) | [diffusion][RFC]Realtime Diffusion Design | IPostYellow | 2026-02-26 | ⚪ 已关闭 | 一般 |  |  |
| [#19487](https://github.com/sgl-project/sglang/issues/19487) | [RFC] [EPLB] Topology-Aware EPLB | Yancey0623 | 2026-02-27 | ⚪ 已关闭 | 一般 |  |  |
| [#19823](https://github.com/sgl-project/sglang/issues/19823) | [Rfc] How to reduce wave of AI spam PRs | vincentzed | 2026-03-04 | ⚪ 已关闭 | 一般 |  |  |
| [#19887](https://github.com/sgl-project/sglang/issues/19887) | [RFC] [Feature] Fast KV-cache pruning (H20 style) | maxim-shevtsov | 2026-03-04 | ⚪ 已关闭 | 一般 |  |  |
| [#20486](https://github.com/sgl-project/sglang/issues/20486) | [RFC] Release cuda 13.0 docker image as default | rainj-me | 2026-03-13 | ⚪ 已关闭 | 一般 |  |  |
| [#22344](https://github.com/sgl-project/sglang/issues/22344) | [RFC]: Internal Process-level Fault Tolerance for SGLang | gaidandawang-afk | 2026-04-08 | ⚪ 已关闭 | 一般 |  |  |
| [#22474](https://github.com/sgl-project/sglang/issues/22474) | [RFC]: Real-Time Streaming Audio Input for ASR Models | SammLSH | 2026-04-10 | ⚪ 已关闭 | 一般 |  |  |
| [#22788](https://github.com/sgl-project/sglang/issues/22788) | [RFC] Elastic EP Scale | zackyoray | 2026-04-14 | ⚪ 已关闭 | 一般 |  |  |
| [#23206](https://github.com/sgl-project/sglang/issues/23206) | [RFC] Sglang non-GPU process rust migration | rainj-me | 2026-04-20 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23206-rfc-sglang-non-gpu-process-rust-migratio.md) |
| [#24625](https://github.com/sgl-project/sglang/issues/24625) | RFC: migrate deepseek_vl2 vision tower from timm to transformers SiglipVisionModel | 1fanwang | 2026-05-07 | ⚪ 已关闭 | 一般 |  |  |
| [#25664](https://github.com/sgl-project/sglang/issues/25664) | Deprecate Hopper SBO feature | b8zhong | 2026-05-18 | ⚪ 已关闭 | 一般 |  |  |
| [#26889](https://github.com/sgl-project/sglang/issues/26889) | [RFC] SGLang IR A Functional Intermediate Representation for SGLang Operators | pre-master | 2026-06-01 | ⚪ 已关闭 | 一般 |  |  |
| [#28585](https://github.com/sgl-project/sglang/issues/28585) | [RFC] Weight loading: separate loading from completeness verification | EazyReal | 2026-06-18 | ⚪ 已关闭 | 一般 |  |  |
| [#29099](https://github.com/sgl-project/sglang/issues/29099) | [RFC] Agent-aware Session-level Cache Preemption for StreamingSession | ykdcai | 2026-06-24 | ⚪ 已关闭 | 一般 |  |  |
| [#29184](https://github.com/sgl-project/sglang/issues/29184) | [RFC] Generalize cuda_graph_* server args / runner naming to a device-agnostic scheme (de… | siju-samuel | 2026-06-24 | ⚪ 已关闭 | 一般 |  |  |
| [#30419](https://github.com/sgl-project/sglang/issues/30419) | [RFC] KVTC KV-cache compression method for SGLang | jfckm | 2026-07-07 | ⚪ 已关闭 | 一般 |  |  |
| [#30696](https://github.com/sgl-project/sglang/issues/30696) | [RFC] RuntimeContext as the configuration API | ch-wan | 2026-07-09 | 🟢 开放 | 一般 |  |  |
| [#31774](https://github.com/sgl-project/sglang/issues/31774) | [RFC] Backend, KV dtype and platform compatibility is checked case by case, and the respo… | mosya415 | 2026-07-20 | ⚪ 已关闭 | 一般 |  |  |
| [#32143](https://github.com/sgl-project/sglang/issues/32143) | [RFC][PD] Seamless P-to-D handoff via token replay | catyans | 2026-07-23 | ⚪ 已关闭 | 一般 |  |  |
| [#32335](https://github.com/sgl-project/sglang/issues/32335) | [RFC]: Native Heterogeneous Weight Conversion in SGLang | Bo-Vincent | 2026-07-24 | ⚪ 已关闭 | 一般 |  |  |
| [#32833](https://github.com/sgl-project/sglang/issues/32833) | [RFC] [MLX] Write down the per-step contract that a replacement event loop must satisfy | jlee5814 | 2026-07-30 | ⚪ 已关闭 | 一般 |  |  |
| [#33409](https://github.com/sgl-project/sglang/issues/33409) | [RFC] Distributed multimodal preprocessing for large multi-image requests in multi-node T… | frankjoey2048 | 2026-08-03 | ⚪ 已关闭 | 一般 |  |  |
| [#35648](https://github.com/sgl-project/sglang/issues/35648) | [Feature] [RFC] Same-GPU model replicas with managed CUDA MPS and a shared KV pool | adityavaid | 2026-08-20 | ⚪ 已关闭 | 一般 |  |  |
| [#35789](https://github.com/sgl-project/sglang/issues/35789) | RFC: Re-layer Draft SWA into Committed Sidecar KV and Verify Scratch | zhangxiaolei123456 | 2026-08-21 | ⚪ 已关闭 | 一般 |  |  |
| [#36208](https://github.com/sgl-project/sglang/issues/36208) | [RFC] Per-Token-Range KV-Cache Retention Priority | limarkdcunha | 2026-08-24 | ⚪ 已关闭 | 一般 |  |  |
| [#36224](https://github.com/sgl-project/sglang/issues/36224) | [RFC] First-Class, Versioned KV Hint Envelope for SGLang | ishandhanani | 2026-08-24 | 🟢 开放 | 一般 |  |  |
| [#37405](https://github.com/sgl-project/sglang/issues/37405) | [RFC] Incremental test cleanup, CI taxonomy, and shared diffusion dispatch | BBuf | 2026-09-01 | ⚪ 已关闭 | 一般 |  |  |
| [#38357](https://github.com/sgl-project/sglang/issues/38357) | [RFC / Feature]: Zero-Copy In-Place KV-Cache Compactor via In-Situ Cycle Permutations (1.… | aemre-cetin | 2026-09-07 | ⚪ 已关闭 | 一般 |  |  |
| [#39963](https://github.com/sgl-project/sglang/issues/39963) | [RFC] Asymmetric P/D deployment for DeepSeek-V4.1 Flash | gz944367214 | 2026-09-17 | 🟢 开放 | 一般 |  |  |
| [#40920](https://github.com/sgl-project/sglang/issues/40920) | [RFC] SGLang renderer | sagearc | 2026-09-23 | 🟢 开放 | 一般 |  |  |
| [#40956](https://github.com/sgl-project/sglang/issues/40956) | [RFC] Some details relates to SGLang renderer item 4 | Edwardssss | 2026-09-23 | 🟢 开放 | 一般 |  |  |
| [#41576](https://github.com/sgl-project/sglang/issues/41576) | [RFC] Dynamic Latent Consensus Governor for DeepSeek-R1 to Reduce KV-Cache Holding Time b… | Arnab758 | 2026-09-28 | 🟢 开放 | 一般 |  |  |
| [#42173](https://github.com/sgl-project/sglang/issues/42173) | RFC: Capability-based device gating for CUDA-compatible OOT platforms | Peter9606 | 2026-10-02 | 🟢 开放 | 一般 |  |  |
| [#42915](https://github.com/sgl-project/sglang/issues/42915) | [RFC] Use reasoning effort as an output-length hint for DP routing and retraction | vincentzed | 2026-10-07 | 🟢 开放 | 一般 |  |  |
| [#44523](https://github.com/sgl-project/sglang/issues/44523) | [Feature] [RFC] Ownership boundaries for externally orchestrated checkpoint and restore | leeZardNav | 2026-10-15 | 🟢 开放 | 一般 |  |  |
| [#45128](https://github.com/sgl-project/sglang/issues/45128) | [RFC] linker: Prefetch SSD-resident KV while the request is queued | alphabetc1 | 2026-10-19 | 🟢 开放 | 一般 |  |  |
| [#45415](https://github.com/sgl-project/sglang/issues/45415) | [RFC] Time-budgeted prefill admission for colocated TP serving | Frank-whw | 2026-10-20 | 🟢 开放 | 一般 |  |  |
| [#45493](https://github.com/sgl-project/sglang/issues/45493) | [RFC] Expose allocator-issued page generations to external linkers | feichai0017 | 2026-10-21 | 🟢 开放 | 一般 |  |  |
| [#46130](https://github.com/sgl-project/sglang/issues/46130) | [RFC] diffusion: LTX-2 stage-1 distilled LoRA base restore is not exact | OnePunchMonk | 2026-10-23 | 🟢 开放 | 一般 |  |  |
| [#46678](https://github.com/sgl-project/sglang/issues/46678) | [RFC] Activation monitors in SGLang | neilkg | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#47170](https://github.com/sgl-project/sglang/issues/47170) | [RFC] Model Independent Parallelism Layer Boundaries and Weight Loading | ch-wan | 2026-10-29 | 🟢 开放 | 一般 |  |  |
| [#47763](https://github.com/sgl-project/sglang/issues/47763) | [RFC] Agent harness x SGLang: keep multi-turn agent prefixes cacheable | alphabetc1 | 2026-11-01 | 🟢 开放 | 一般 |  |  |
| [#48300](https://github.com/sgl-project/sglang/issues/48300) | [Feature][RFC] Preserve measured cache-write usage across PD handoff | YaoleWu | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#48344](https://github.com/sgl-project/sglang/issues/48344) | [RFC] mem_cache: Cross-branch KV page splicing and cursor fork/rewind for hybrid models (… | shatianming5 | 2026-11-04 | ⚪ 已关闭 | 一般 |  |  |

## 数据快照(下次更新的基线)

- 快照记录时间: 2026-11-07T13:53:26+00:00
- **vllm** @1ecc9cc4ce66 (upstream HEAD 2026-11-07) — commit: https://github.com/vllm-project/vllm/commit/1ecc9cc4ce6626352df39e4364579ec376c10761 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 917 → 重点档 222 → 摘要卡 222
- **sglang** @ba5906471c62 (upstream HEAD 2026-11-07) — commit: https://github.com/sgl-project/sglang/commit/ba5906471c62e82c0594a617b4d07d9b2c0fd2b3 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 202 → 重点档 42 → 摘要卡 42
