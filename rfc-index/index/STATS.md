# RFC 索引统计总览

- 生成时间: 2026-11-08T02:05:32+00:00

## 总量

| 仓库 | RFC 总数 | 🔥 重点 | 开放 | 已关闭 | 有收益声明 |
| --- | --- | --- | --- | --- | --- |
| vllm | 917 | 222 | 371 | 546 | 24 |
| sglang | 202 | 42 | 68 | 134 | 2 |

## vllm:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 53 | 48 |
| MoE(混合专家) | 38 | 12 |
| Quantization(量化) | 43 | 11 |
| Disaggregated Serving(分离式部署) | 53 | 25 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 33 | 14 |
| Overlap / Scheduler(重叠与调度) | 28 | 4 |
| Attention / Kernel(注意力与算子) | 56 | 26 |
| KV Cache / Memory(KV Cache 与显存) | 80 | 9 |
| Parallelism / Communication(并行与通信) | 19 | 2 |
| Sampler / Detokenization(采样与反分词) | 14 | 0 |
| Frontend / Serving / API(前端与服务 API) | 49 | 3 |
| Architecture / Engine(架构与引擎) | 51 | 9 |
| Hardware / Backend(硬件与后端) | 27 | 5 |
| Feature / Ecosystem(功能与生态) | 81 | 14 |
| CI / Infra / Process(CI 与基础流程) | 25 | 8 |
| Other(其他) | 267 | 32 |

## vllm:年度分布(按创建时间)

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 569 | 123 |
| 2025 | 252 | 64 |
| 2024 | 94 | 34 |
| 2023 | 2 | 1 |

## vllm:热度 Top 20 (reactions×2 + comments)

| 热度 | RFC | 标题 | 状态 |
| --- | --- | --- | --- |
| 210 | [#4194](https://github.com/vllm-project/vllm/issues/4194) | [RFC]: Multi-modality Support on vLLM | 🟢 开放 |
| 197 | [#18571](https://github.com/vllm-project/vllm/issues/18571) | [RFC]: Deprecating vLLM V0 | ⚪ 已关闭 |
| 194 | [#8779](https://github.com/vllm-project/vllm/issues/8779) | vLLM's V1 Engine Architecture | 🟢 开放 |
| 188 | [#40846](https://github.com/vllm-project/vllm/issues/40846) | [RFC]: Rust front-end | 🟢 开放 |
| 146 | [#6226](https://github.com/vllm-project/vllm/issues/6226) | [RFC] Drop beam search support | ⚪ 已关闭 |
| 135 | [#2614](https://github.com/vllm-project/vllm/issues/2614) | [RFC] Automatic Prefix Caching | ⚪ 已关闭 |
| 104 | [#42770](https://github.com/vllm-project/vllm/issues/42770) | [RFC]: Changes in vLLM Model Development | 🟢 开放 |
| 95 | [#5557](https://github.com/vllm-project/vllm/issues/5557) | [RFC]: Implement disaggregated prefilling via KV cache tran… | ⚪ 已关闭 |
| 95 | [#19854](https://github.com/vllm-project/vllm/issues/19854) | [RFC]: KV cache offloading | 🟢 开放 |
| 94 | [#34303](https://github.com/vllm-project/vllm/issues/34303) | [RFC]: CUDA Checkpoint/Restore for Near-Zero Cold Starts | 🟢 开放 |
| 92 | [#44219](https://github.com/vllm-project/vllm/issues/44219) | [RFC]: hardware agnostic model definitions in vLLM | 🟢 开放 |
| 89 | [#3620](https://github.com/vllm-project/vllm/issues/3620) | [RFC] Initial Support for Cloud TPUs | ⚪ 已关闭 |
| 86 | [#4565](https://github.com/vllm-project/vllm/issues/4565) | [RFC]: Automate Speculative Decoding | ⚪ 已关闭 |
| 81 | [#10818](https://github.com/vllm-project/vllm/issues/10818) | [RFC]: Disaggregated prefilling and KV cache transfer roadm… | ⚪ 已关闭 |
| 80 | [#7692](https://github.com/vllm-project/vllm/issues/7692) | [RFC]: Add Ascend NPU as a new backend | ⚪ 已关闭 |
| 76 | [#40902](https://github.com/vllm-project/vllm/issues/40902) | [Roadmap] DeepSeek V4 | ⚪ 已关闭 |
| 72 | [#6854](https://github.com/vllm-project/vllm/issues/6854) | [RFC]: Multi-Step Scheduling | ⚪ 已关闭 |
| 69 | [#22932](https://github.com/vllm-project/vllm/issues/22932) | [RFC]: Path to vLLM 1.0 Release | 🟢 开放 |
| 67 | [#22992](https://github.com/vllm-project/vllm/issues/22992) | [RFC]: Refactor CI/CD | ⚪ 已关闭 |
| 66 | [#19038](https://github.com/vllm-project/vllm/issues/19038) | [RFC]: Prefill-only optimizations for PD disaggregation in … | 🟢 开放 |

## vllm:Top 作者 (前 15)

| 作者 | RFC 数 |
| --- | --- |
| [aoshen02](https://github.com/aoshen02) | 27 |
| [ProExpertProg](https://github.com/ProExpertProg) | 17 |
| [youkaichao](https://github.com/youkaichao) | 11 |
| [WoosukKwon](https://github.com/WoosukKwon) | 9 |
| [simon-mo](https://github.com/simon-mo) | 9 |
| [robertgshaw2-redhat](https://github.com/robertgshaw2-redhat) | 9 |
| [LucasWilkinson](https://github.com/LucasWilkinson) | 9 |
| [zou3519](https://github.com/zou3519) | 9 |
| [algebra-MCX](https://github.com/algebra-MCX) | 9 |
| [tjtanaa](https://github.com/tjtanaa) | 8 |
| [pisceskkk](https://github.com/pisceskkk) | 7 |
| [afeldman-nm](https://github.com/afeldman-nm) | 6 |
| [russellb](https://github.com/russellb) | 6 |
| [DarkLight1337](https://github.com/DarkLight1337) | 6 |
| [wangxiyuan](https://github.com/wangxiyuan) | 6 |

## sglang:分类分布

| 分类 | 总数 | 🔥 重点 |
| --- | --- | --- |
| Speculative Decoding(投机解码) | 9 | 6 |
| MoE(混合专家) | 6 | 1 |
| Quantization(量化) | 12 | 2 |
| Disaggregated Serving(分离式部署) | 12 | 10 |
| CUDA Graph / Compiler(CUDA Graph 与编译) | 4 | 4 |
| Overlap / Scheduler(重叠与调度) | 11 | 0 |
| Attention / Kernel(注意力与算子) | 13 | 7 |
| KV Cache / Memory(KV Cache 与显存) | 22 | 3 |
| Frontend / Serving / API(前端与服务 API) | 13 | 0 |
| Architecture / Engine(架构与引擎) | 15 | 2 |
| Hardware / Backend(硬件与后端) | 8 | 1 |
| Feature / Ecosystem(功能与生态) | 18 | 0 |
| CI / Infra / Process(CI 与基础流程) | 4 | 0 |
| Other(其他) | 55 | 6 |

## sglang:年度分布(按创建时间)

| 年份 | 总数 | 🔥 重点 |
| --- | --- | --- |
| 2026 | 171 | 30 |
| 2025 | 31 | 12 |

## sglang:热度 Top 20 (reactions×2 + comments)

| 热度 | RFC | 标题 | 状态 |
| --- | --- | --- | --- |
| 117 | [#16546](https://github.com/sgl-project/sglang/issues/16546) | [RFC] SGLang-Omni Design | ⚪ 已关闭 |
| 103 | [#12766](https://github.com/sgl-project/sglang/issues/12766) | [Feature] RFC: Block Diffusion Large Language Model (dLLM) … | ⚪ 已关闭 |
| 52 | [#7746](https://github.com/sgl-project/sglang/issues/7746) | [RFC] Remote KV Connector for SGLang Global Cache Reuse and… | ⚪ 已关闭 |
| 43 | [#7851](https://github.com/sgl-project/sglang/issues/7851) | [RFC] DRI for every module | ⚪ 已关闭 |
| 42 | [#2807](https://github.com/sgl-project/sglang/issues/2807) | [Feature] RFC for adding CPU support for SGLang | ⚪ 已关闭 |
| 41 | [#29630](https://github.com/sgl-project/sglang/issues/29630) | [RFC] Introduce a unified sglang.kernels namespace for kern… | ⚪ 已关闭 |
| 39 | [#23206](https://github.com/sgl-project/sglang/issues/23206) | [RFC] Sglang non-GPU process rust migration | 🟢 开放 |
| 37 | [#24656](https://github.com/sgl-project/sglang/issues/24656) | [RFC] Agent-Aware KV Cache Phase 1 for Agentic Workloads | 🟢 开放 |
| 33 | [#7332](https://github.com/sgl-project/sglang/issues/7332) | [RFC] Bi-weekly release | ⚪ 已关闭 |
| 28 | [#22044](https://github.com/sgl-project/sglang/issues/22044) | [Feature][RFC] SPECTRE: Parallel Speculative Decoding with … | ⚪ 已关闭 |
| 27 | [#24945](https://github.com/sgl-project/sglang/issues/24945) | [RFC] SGLang EPD Performance / Architecture / Observability… | ⚪ 已关闭 |
| 26 | [#19421](https://github.com/sgl-project/sglang/issues/19421) | [diffusion][RFC]Realtime Diffusion Design | ⚪ 已关闭 |
| 25 | [#9853](https://github.com/sgl-project/sglang/issues/9853) | RFC: Command-Line Interface Architecture for SGLang | ⚪ 已关闭 |
| 23 | [#22558](https://github.com/sgl-project/sglang/issues/22558) | [RFC] Native gRPC Server for SGLang in Rust | ⚪ 已关闭 |
| 22 | [#32321](https://github.com/sgl-project/sglang/issues/32321) | [RFC] Apple Silicon serving redesign: Torch-owned SRT path … | 🟢 开放 |
| 21 | [#5471](https://github.com/sgl-project/sglang/issues/5471) | [RFC][Feature][Model] Add templated fallback HF `transforme… | ⚪ 已关闭 |
| 21 | [#11334](https://github.com/sgl-project/sglang/issues/11334) | [RFC] Directory Structure Does Not Align with Non-PyTorch I… | ⚪ 已关闭 |
| 21 | [#30928](https://github.com/sgl-project/sglang/issues/30928) | [RFC] Position-Independent KV Cache Reuse for Agentic/RAG W… | ⚪ 已关闭 |
| 19 | [#7427](https://github.com/sgl-project/sglang/issues/7427) | 🚧  RFC: Redesign Batch Processing as an Offline Workflow | ⚪ 已关闭 |
| 18 | [#10464](https://github.com/sgl-project/sglang/issues/10464) | [Feature] RFC: Integrating checkpoint engine into SGLang | ⚪ 已关闭 |

## sglang:Top 作者 (前 15)

| 作者 | RFC 数 |
| --- | --- |
| [b8zhong](https://github.com/b8zhong) | 7 |
| [vincentzed](https://github.com/vincentzed) | 6 |
| [zhyncs](https://github.com/zhyncs) | 4 |
| [alphabetc1](https://github.com/alphabetc1) | 4 |
| [rainj-me](https://github.com/rainj-me) | 3 |
| [stmatengss](https://github.com/stmatengss) | 3 |
| [sagearc](https://github.com/sagearc) | 3 |
| [CatherineSue](https://github.com/CatherineSue) | 2 |
| [DevashishLal-CB](https://github.com/DevashishLal-CB) | 2 |
| [nvjullin](https://github.com/nvjullin) | 2 |
| [pre-master](https://github.com/pre-master) | 2 |
| [chenxb002](https://github.com/chenxb002) | 2 |
| [BBuf](https://github.com/BBuf) | 2 |
| [anencore94](https://github.com/anencore94) | 2 |
| [ch-wan](https://github.com/ch-wan) | 2 |

## 数据快照(下次更新的基线)

- 快照记录时间: 2026-11-07T13:53:26+00:00
- **vllm** @1ecc9cc4ce66 (upstream HEAD 2026-11-07) — commit: https://github.com/vllm-project/vllm/commit/1ecc9cc4ce6626352df39e4364579ec376c10761 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 917 → 重点档 222 → 摘要卡 222
- **sglang** @ba5906471c62 (upstream HEAD 2026-11-07) — commit: https://github.com/sgl-project/sglang/commit/ba5906471c62e82c0594a617b4d07d9b2c0fd2b3 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 202 → 重点档 42 → 摘要卡 42
