# VLLM RFC 索引(label:rfc ∪ 标题含 RFC,全量)

- 仓库: https://github.com/vllm-project/vllm
- 收录总数: **917** (重点 🔥 **222** / 一般 695) | 开放 371 / 已关闭 546
- 生成时间: 2026-11-08T02:05:31+00:00

## 目录

- [Speculative Decoding(投机解码) (53)](#speculative-decoding投机解码-53)
- [MoE(混合专家) (38)](#moe混合专家-38)
- [Quantization(量化) (43)](#quantization量化-43)
- [Disaggregated Serving(分离式部署) (53)](#disaggregated-serving分离式部署-53)
- [CUDA Graph / Compiler(CUDA Graph 与编译) (33)](#cuda-graph--compilercuda-graph-与编译-33)
- [Overlap / Scheduler(重叠与调度) (28)](#overlap--scheduler重叠与调度-28)
- [Attention / Kernel(注意力与算子) (56)](#attention--kernel注意力与算子-56)
- [KV Cache / Memory(KV Cache 与显存) (80)](#kv-cache--memorykv-cache-与显存-80)
- [Parallelism / Communication(并行与通信) (19)](#parallelism--communication并行与通信-19)
- [Sampler / Detokenization(采样与反分词) (14)](#sampler--detokenization采样与反分词-14)
- [Frontend / Serving / API(前端与服务 API) (49)](#frontend--serving--api前端与服务-api-49)
- [Architecture / Engine(架构与引擎) (51)](#architecture--engine架构与引擎-51)
- [Hardware / Backend(硬件与后端) (27)](#hardware--backend硬件与后端-27)
- [Feature / Ecosystem(功能与生态) (81)](#feature--ecosystem功能与生态-81)
- [CI / Infra / Process(CI 与基础流程) (25)](#ci--infra--processci-与基础流程-25)
- [Other(其他) (267)](#other其他-267)

## Speculative Decoding(投机解码) (53)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#3809](https://github.com/vllm-project/vllm/issues/3809) | [RFC] Augment BaseExecutor interfaces to enable hardware-agnostic speculative decoding | cadedaniel | 2024-04-03 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-3809-rfc-augment-baseexecutor-interfaces-to-e.md) |
| [#4565](https://github.com/vllm-project/vllm/issues/4565) | [RFC]: Automate Speculative Decoding | LiuXiaoxuanPKU | 2024-05-02 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-4565-rfc-automate-speculative-decoding.md) |
| [#7351](https://github.com/vllm-project/vllm/issues/7351) | Create speculative decode dynamic parallel strategy | vladislavkruglikov | 2024-08-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7351-create-speculative-decode-dynamic-parall.md) |
| [#14719](https://github.com/vllm-project/vllm/issues/14719) | [RFC][V1][Spec Decode] V1 Spec Decode Eagle Support | LiuXiaoxuanPKU | 2025-03-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-14719-rfc-v1-spec-decode-v1-spec-decode-eagle.md) |
| [#18633](https://github.com/vllm-project/vllm/issues/18633) | [RFC]: [Spec Decode] Combine Ngram and EAGLE | ekagra-ranjan | 2025-05-23 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18633-rfc-spec-decode-combine-ngram-and-eagle.md) |
| [#22547](https://github.com/vllm-project/vllm/issues/22547) | [RFC]: vllm 0.10.1gpt-oss version, will speculative sampling be used by default? | 123wujiao | 2025-08-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22547-rfc-vllm-0-10-1gpt-oss-version-will-spec.md) |
| [#22549](https://github.com/vllm-project/vllm/issues/22549) | [RFC]: vllm 0.10.1gpt-oss version, will speculative sampling be used by default? | 123wujiao | 2025-08-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22549-rfc-vllm-0-10-1gpt-oss-version-will-spec.md) |
| [#22550](https://github.com/vllm-project/vllm/issues/22550) | [RFC]: vllm 0.10.1gpt-oss，speculative sampling be use？ | 123wujiao | 2025-08-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22550-rfc-vllm-0-10-1gpt-oss-speculative-sampl.md) |
| [#28135](https://github.com/vllm-project/vllm/issues/28135) | [RFC]: Robust End-to-End CI/CD Regression Testing for Speculative Decoding | rahul-tuli | 2025-11-05 | ⚪ 已关闭 | 🔥 重点 | Speculative decoding improves LLM latency 2-3x by using a d… | [卡片](rfc-cards/rfc-28135-rfc-robust-end-to-end-ci-cd-regression-t.md) |
| [#31204](https://github.com/vllm-project/vllm/issues/31204) | [RFC]: Supporting Multi MTP layers in Speculative Decoding (EagleProposer) | DingYiBin | 2025-12-23 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-31204-rfc-supporting-multi-mtp-layers-in-specu.md) |
| [#31483](https://github.com/vllm-project/vllm/issues/31483) | [RFC]: a efficient adaptive rejection sampling for accelerating speculative decoding. | sunchendd | 2025-12-29 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-31483-rfc-a-efficient-adaptive-rejection-sampl.md) |
| [#36219](https://github.com/vllm-project/vllm/issues/36219) | [RFC]: Speculative Decoding Proposer Interface Unification Proposal | wangxiyuan | 2026-03-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-36219-rfc-speculative-decoding-proposer-interf.md) |
| [#36223](https://github.com/vllm-project/vllm/issues/36223) | [RFC]: Support quarot for eagle3 | drslark | 2026-03-06 | ⚪ 已关闭 | 一般 |  |  |
| [#36657](https://github.com/vllm-project/vllm/issues/36657) | [RFC]: Dynamic Speculation Length (DSL) with Confidence-Threshold Early Exit for vLLM Spe… | jmamou | 2026-03-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-36657-rfc-dynamic-speculation-length-dsl-with.md) |
| [#38003](https://github.com/vllm-project/vllm/issues/38003) | [RFC]: Integrate MineDraft speculative decoding into vLLM | Sebastian-dong | 2026-03-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-38003-rfc-integrate-minedraft-speculative-deco.md) |
| [#38069](https://github.com/vllm-project/vllm/issues/38069) | [RFC]: Can we implement n-gram and suffix speculative decoding in model_runner_v2? | lio1226 | 2026-03-25 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-38069-rfc-can-we-implement-n-gram-and-suffix-s.md) |
| [#42109](https://github.com/vllm-project/vllm/issues/42109) | [RFC]: Disaggregated Speculative Decoding with Standalone Parallel Draft Model | laviier | 2026-05-08 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42109-rfc-disaggregated-speculative-decoding-w.md) |
| [#43545](https://github.com/vllm-project/vllm/issues/43545) | [RFC]: Add Gumiho speculative decoding to vLLM | userljz | 2026-05-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-43545-rfc-add-gumiho-speculative-decoding-to-v.md) |
| [#44697](https://github.com/vllm-project/vllm/issues/44697) | [RFC]: MTP speculative decoding under pipeline parallelism (PP>1) | atassis | 2026-06-05 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-44697-rfc-mtp-speculative-decoding-under-pipel.md) |
| [#44790](https://github.com/vllm-project/vllm/issues/44790) | [RFC]: Support Bailing MTP (Multi-Token Prediction) for Ling-2.6-flash | ghost | 2026-06-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-44790-rfc-support-bailing-mtp-multi-token-pred.md) |
| [#44826](https://github.com/vllm-project/vllm/issues/44826) | [RFC]: MTP Routing for Qwen3.5 Series Multi-LoRA Deployments | crossxxd | 2026-06-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-44826-rfc-mtp-routing-for-qwen3-5-series-multi.md) |
| [#46707](https://github.com/vllm-project/vllm/issues/46707) | [RFC]: Support for Domino speculative decoding in vLLM | GIREESH7963 | 2026-06-25 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-46707-rfc-support-for-domino-speculative-decod.md) |
| [#47351](https://github.com/vllm-project/vllm/issues/47351) | [RFC]: Sparse-attention-based self-speculative decoding (StreamingLLM, then Vegas) | npz7yyk | 2026-07-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47351-rfc-sparse-attention-based-self-speculat.md) |
| [#47572](https://github.com/vllm-project/vllm/issues/47572) | [RFC]: ReplaySSM: cache SSM inputs instead of state for faster standard and speculative d… | Johnny-Liou | 2026-07-03 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47572-rfc-replayssm-cache-ssm-inputs-instead-o.md) |
| [#47839](https://github.com/vllm-project/vllm/issues/47839) | [RFC]: Packed Variable Length Speculative Decoding | benchislett | 2026-07-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47839-rfc-packed-variable-length-speculative-d.md) |
| [#48197](https://github.com/vllm-project/vllm/issues/48197) | [RFC]: StructuredOutputManager x Speculative Decoding Refactor | yzong-rh | 2026-07-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48197-rfc-structuredoutputmanager-x-speculativ.md) |
| [#48202](https://github.com/vllm-project/vllm/issues/48202) | [RFC]: Per-request effective proposal lengths for adaptive speculative decoding | LJX-xixi | 2026-07-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48202-rfc-per-request-effective-proposal-lengt.md) |
| [#48296](https://github.com/vllm-project/vllm/issues/48296) | [RFC] Speculative Decoding Lifecycle for RL Rollout | aoshen02 | 2026-07-11 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48296-rfc-speculative-decoding-lifecycle-for-r.md) |
| [#48445](https://github.com/vllm-project/vllm/issues/48445) | [RFC]: Support SparDA-style decoupled Forecast projection for lookahead KV prefetch in sp… | Srinivasoo7 | 2026-07-13 | 🟢 开放 | 一般 |  |  |
| [#48627](https://github.com/vllm-project/vllm/issues/48627) | [RFC]: Context-length-aware speculative token scheduling — extending num_speculative_toke… | seongyun1104 | 2026-07-14 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48627-rfc-context-length-aware-speculative-tok.md) |
| [#49090](https://github.com/vllm-project/vllm/issues/49090) | [RFC][SpecDecode] Move MTP completeness validation to the weight-update transaction bound… | aoshen02 | 2026-07-19 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49090-rfc-specdecode-move-mtp-completeness-val.md) |
| [#49232](https://github.com/vllm-project/vllm/issues/49232) | [RFC]: ReplaySSM: cache SSM inputs instead of state for faster standard and speculative d… | askliar | 2026-07-20 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49232-rfc-replayssm-cache-ssm-inputs-instead-o.md) |
| [#49488](https://github.com/vllm-project/vllm/issues/49488) | [RFC] Persistent, in-place drafting attention metadata for full-CUDA-graph MTP/spec-decode | chanh | 2026-07-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49488-rfc-persistent-in-place-drafting-attenti.md) |
| [#49545](https://github.com/vllm-project/vllm/issues/49545) | [RFC]: Speculative recompute fallback for slow remote KV cache loads | Lin-z-w | 2026-07-23 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49545-rfc-speculative-recompute-fallback-for-s.md) |
| [#50391](https://github.com/vllm-project/vllm/issues/50391) | [RFC]: Support Speculative Decoding with Decode Context Parallelism | pisceskkk | 2026-07-30 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-50391-rfc-support-speculative-decoding-with-de.md) |
| [#50438](https://github.com/vllm-project/vllm/issues/50438) | [RFC]: Lookahead-aware prefix-cache hashing for EAGLE-style draft models | skajre | 2026-07-30 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-50438-rfc-lookahead-aware-prefix-cache-hashing.md) |
| [#50853](https://github.com/vllm-project/vllm/issues/50853) | [RFC]: Complete Model Runner V2 pipeline parallelism: scheduling, speculative decoding, a… | LostFox11 | 2026-08-03 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-50853-rfc-complete-model-runner-v2-pipeline-pa.md) |
| [#52038](https://github.com/vllm-project/vllm/issues/52038) | [RFC]: LoRA adapter support for DFlash speculative decoding draft models | anshkaggarwal22 | 2026-08-12 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-52038-rfc-lora-adapter-support-for-dflash-spec.md) |
| [#53673](https://github.com/vllm-project/vllm/issues/53673) | [RFC]: Decouple draft-model parallelism from the target model (PCP/DCP + speculative deco… | LucasWilkinson | 2026-08-25 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-53673-rfc-decouple-draft-model-parallelism-fro.md) |
| [#54080](https://github.com/vllm-project/vllm/issues/54080) | [RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models | sneha5gsm | 2026-08-27 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-54080-rfc-treewy-tree-speculative-decoding-for.md) |
| [#54506](https://github.com/vllm-project/vllm/issues/54506) | [RFC]: Batch invariance for speculative decoding needs to cover the forward pass (M=1 vs … | Leonccaa | 2026-08-31 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-54506-rfc-batch-invariance-for-speculative-dec.md) |
| [#54661](https://github.com/vllm-project/vllm/issues/54661) | [RFC]: Per-KV-cache-group prefix-cache retention, with a hit-min exemption for hit-inert … | nood-co1 | 2026-09-01 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-54661-rfc-per-kv-cache-group-prefix-cache-rete.md) |
| [#56701](https://github.com/vllm-project/vllm/issues/56701) | [RFC][KV Offload]: Align sliding-window restore coverage with MTP-retained history | jacklin78911-collab | 2026-09-13 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-56701-rfc-kv-offload-align-sliding-window-rest.md) |
| [#56880](https://github.com/vllm-project/vllm/issues/56880) | [RFC]: Add NCP-OLMo support and NCP DFlash speculative decoding | LuckySJTU | 2026-09-14 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-56880-rfc-add-ncp-olmo-support-and-ncp-dflash.md) |
| [#58638](https://github.com/vllm-project/vllm/issues/58638) | [RFC]: KV cache grouping for hybrid models with speculative drafters | LucasWilkinson | 2026-09-25 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-58638-rfc-kv-cache-grouping-for-hybrid-models.md) |
| [#59117](https://github.com/vllm-project/vllm/issues/59117) | [RFC]: [Spec decode] KV-cache pool collapse when a draft model introduces a new cache spe… | ZZXX-bit | 2026-09-29 | 🟢 开放 | 一般 |  |  |
| [#61761](https://github.com/vllm-project/vllm/issues/61761) | [RFC]: DFlash2 speculative decoding performance on ROCm (MI355X) | zhentaocc | 2026-10-15 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-61761-rfc-dflash2-speculative-decoding-perform.md) |
| [#62640](https://github.com/vllm-project/vllm/issues/62640) | [RFC]: Improve MTP configuration warning | nfm-corti | 2026-10-20 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-62640-rfc-improve-mtp-configuration-warning.md) |
| [#62818](https://github.com/vllm-project/vllm/issues/62818) | [RFC]: Preserve output distribution under structured-output + speculative decoding | yzong-rh | 2026-10-21 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-62818-rfc-preserve-output-distribution-under-s.md) |
| [#63666](https://github.com/vllm-project/vllm/issues/63666) | [RFC]: Tiered MoE Expert Offload with Lookahead Expert Scheduling (LES) for single-GPU In… | wyc55069407 | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#64002](https://github.com/vllm-project/vllm/issues/64002) | [RFC] [DSV4.1] [Engram]: Share immutable host-backed Engram tables across independent ins… | foraxe | 2026-10-27 | 🟢 开放 | 一般 |  |  |
| [#65119](https://github.com/vllm-project/vllm/issues/65119) | [RFC]: DBO for the MTP/EAGLE drafter forward in Model Runner V2 | algebra-MCX | 2026-11-02 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-65119-rfc-dbo-for-the-mtp-eagle-drafter-forwar.md) |
| [#65813](https://github.com/vllm-project/vllm/issues/65813) | [RFC]: Add a Tree Speculative Decoding Framework | drslark | 2026-11-05 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-65813-rfc-add-a-tree-speculative-decoding-fram.md) |

## MoE(混合专家) (38)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5346](https://github.com/vllm-project/vllm/issues/5346) | [RFC]: Refactor MoE | robertgshaw2-redhat | 2024-06-07 | ⚪ 已关闭 | 一般 |  |  |
| [#9203](https://github.com/vllm-project/vllm/issues/9203) | [RFC]: Shared Mixture of Experts | followumesh | 2024-10-09 | ⚪ 已关闭 | 一般 |  |  |
| [#16037](https://github.com/vllm-project/vllm/issues/16037) | [RFC]: Data Parallel Attention and Expert Parallel MoEs | tlrmchlsmth | 2025-04-03 | ⚪ 已关闭 | 一般 |  |  |
| [#17087](https://github.com/vllm-project/vllm/issues/17087) | [RFC]: Expert parallelism in VLLM - do you do local dropping on sub-batch of token activa… | Neo9061 | 2025-04-24 | ⚪ 已关闭 | 一般 |  |  |
| [#20323](https://github.com/vllm-project/vllm/issues/20323) | [RFC]: Elastic Expert Parallelism | ruisearch42 | 2025-07-01 | 🟢 开放 | 一般 |  |  |
| [#22246](https://github.com/vllm-project/vllm/issues/22246) | [RFC]: Dynamic Expert Load Balance with Zero-like-overhead | raindaywhu | 2025-08-05 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22246-rfc-dynamic-expert-load-balance-with-zer.md) |
| [#22799](https://github.com/vllm-project/vllm/issues/22799) | [RFC]: ATTN-FFN Disaggregation for MoE Models | chopper0126 | 2025-08-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22799-rfc-attn-ffn-disaggregation-for-moe-mode.md) |
| [#24112](https://github.com/vllm-project/vllm/issues/24112) | [RFC]:  Improve MoE triton kernel tuning | jeejeelee | 2025-09-02 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-24112-rfc-improve-moe-triton-kernel-tuning.md) |
| [#26108](https://github.com/vllm-project/vllm/issues/26108) | [RFC]: MoE shared_experts fusion with the selected experts inside FusedMoE | alexm-redhat | 2025-10-02 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-26108-rfc-moe-shared-experts-fusion-with-the-s.md) |
| [#27774](https://github.com/vllm-project/vllm/issues/27774) | [RFC]:  Fault-Tolerant Expert Parallelism | tzulingk | 2025-10-29 | ⚪ 已关闭 | 🔥 重点 | - Detect failures when latency exceeds 3x (configurable thr… | [卡片](rfc-cards/rfc-27774-rfc-fault-tolerant-expert-parallelism.md) |
| [#27908](https://github.com/vllm-project/vllm/issues/27908) | [RFC]: Fault-Tolerant Expert parallelism phase 2 (recovery) | tzulingk | 2025-11-01 | ⚪ 已关闭 | 一般 |  |  |
| [#28192](https://github.com/vllm-project/vllm/issues/28192) | [RFC]: Support separate NICs for KV cache traffic and MoE traffic | JayFzh | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#33549](https://github.com/vllm-project/vllm/issues/33549) | [RFC]: Dynamic Expert Load Balance optimization | shenchuxiaofugui | 2026-02-02 | ⚪ 已关闭 | 一般 |  |  |
| [#33869](https://github.com/vllm-project/vllm/issues/33869) | [RFC]:DeepSeek-R1 Moe offload | wangyxbh | 2026-02-05 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33869-rfc-deepseek-r1-moe-offload.md) |
| [#34963](https://github.com/vllm-project/vllm/issues/34963) | [RFC]: Adding Support for Single Batch Overlap (SBO) With FlashInfer DeepEP LL NVFP4 | elvircrn | 2026-02-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-34963-rfc-adding-support-for-single-batch-over.md) |
| [#35550](https://github.com/vllm-project/vllm/issues/35550) | [RFC]: Batch-Aware Expert Pruning for MoE Decode (XShare) | hai-meh-cs | 2026-02-27 | ⚪ 已关闭 | 一般 |  |  |
| [#38079](https://github.com/vllm-project/vllm/issues/38079) | [RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop | junjzhang | 2026-03-25 | ⚪ 已关闭 | 一般 |  |  |
| [#38256](https://github.com/vllm-project/vllm/issues/38256) | [RFC]: Incremental MoE Expert Offloading — GPU Cache + Async Pipeline | e1n00r | 2026-03-26 | 🟢 开放 | 一般 |  |  |
| [#41204](https://github.com/vllm-project/vllm/issues/41204) | [RFC]: Expert Weight Backup for Elastic EP | jeffreywang88 | 2026-04-29 | ⚪ 已关闭 | 一般 |  |  |
| [#45154](https://github.com/vllm-project/vllm/issues/45154) | [Rust Frontend][RFC]: Elastic Expert Parallel support | coder3101 | 2026-06-10 | ⚪ 已关闭 | 一般 |  |  |
| [#46295](https://github.com/vllm-project/vllm/issues/46295) | [RFC]: Adaptive Speculation Depth for MoE Models via Per-Iteration Utility Budgeting | Mattral | 2026-06-21 | 🟢 开放 | 一般 |  |  |
| [#51005](https://github.com/vllm-project/vllm/issues/51005) | [RFC]: Let Platforms customize fused MoE Router construction | freyfwt | 2026-08-04 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-51005-rfc-let-platforms-customize-fused-moe-ro.md) |
| [#53629](https://github.com/vllm-project/vllm/issues/53629) | [RFC]: Apply MoE weight transformations at disk streaming time instead of materializing r… | wjabbour | 2026-08-24 | 🟢 开放 | 一般 |  |  |
| [#55339](https://github.com/vllm-project/vllm/issues/55339) | [RFC]: Unify ModelOpt MoE methods (mirror of #49381) | ima-helikoptaaa | 2026-09-04 | 🟢 开放 | 一般 |  |  |
| [#56402](https://github.com/vllm-project/vllm/issues/56402) | [RFC]: Efficient Routed-Expert Replay with Prefix Omission and KV Cache Offloading | cccyy123 | 2026-09-11 | 🟢 开放 | 一般 |  |  |
| [#56745](https://github.com/vllm-project/vllm/issues/56745) | [RFC]: Return `routed_experts` on the terminal streaming chunk | ryanliang10000-cmyk | 2026-09-14 | 🟢 开放 | 一般 |  |  |
| [#57794](https://github.com/vllm-project/vllm/issues/57794) | [RFC]: Expert-granular MoE residency — drive UVA offload from EPLB's per-expert load stat… | Mohammad-Nabulsi | 2026-09-20 | 🟢 开放 | 一般 |  |  |
| [#60932](https://github.com/vllm-project/vllm/issues/60932) | [RFC]: Peer-device expert tier: run a subset of routed experts on a second GPU in the sam… | sjronan | 2026-10-10 | 🟢 开放 | 一般 |  |  |
| [#60935](https://github.com/vllm-project/vllm/issues/60935) | [Draft][RFC][MoE]: Support dispatcher-native token dropping for MoE expert capacity | bnellnm | 2026-10-10 | 🟢 开放 | 一般 |  |  |
| [#61539](https://github.com/vllm-project/vllm/issues/61539) | [RFC]: Explicit row layouts for routed-expert capture | Liamup777 | 2026-10-14 | 🟢 开放 | 一般 |  |  |
| [#63650](https://github.com/vllm-project/vllm/issues/63650) | [RFC] M-gated HPC-Ops dual-BF16 router GEMM for fp32 MoE router weights on SM90 | new-TonyWang | 2026-10-26 | ⚪ 已关闭 | 一般 |  |  |
| [#64383](https://github.com/vllm-project/vllm/issues/64383) | [RFC]: R3 as a KV cache layer: where routed-experts rows of prefix-cached blocks live | aoshen02 | 2026-10-29 | 🟢 开放 | 一般 |  |  |
| [#64776](https://github.com/vllm-project/vllm/issues/64776) | [RFC]: Single Batch Overlap (down GEMM ↔ combine) for DeepEP LL | algebra-MCX | 2026-10-30 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-64776-rfc-single-batch-overlap-down-gemm-combi.md) |
| [#64780](https://github.com/vllm-project/vllm/issues/64780) | [RFC]: DBO support for FlashInfer NVLink all-to-all backends (single-node H200/B200 first) | algebra-MCX | 2026-10-30 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-64780-rfc-dbo-support-for-flashinfer-nvlink-al.md) |
| [#65120](https://github.com/vllm-project/vllm/issues/65120) | [RFC]: DBO for the decoder of multimodal MoE models in Model Runner V2 | algebra-MCX | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#65122](https://github.com/vllm-project/vllm/issues/65122) | [RFC]: Zero-copy DeepEP low-latency combine, safe under DBO | algebra-MCX | 2026-11-02 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-65122-rfc-zero-copy-deepep-low-latency-combine.md) |
| [#65209](https://github.com/vllm-project/vllm/issues/65209) | [RFC]: Group-wide LoRA admission and residency for distributed MoE serving | MagellaX | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#66163](https://github.com/vllm-project/vllm/issues/66163) | [RFC][ROCm]: Batch-invariant kernels for Kimi-K3.1 on MI355X (KDA, DSA/indexer, MXFP4 MoE… | indianspeedster | 2026-11-06 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-66163-rfc-rocm-batch-invariant-kernels-for-kim.md) |

## Quantization(量化) (43)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#3975](https://github.com/vllm-project/vllm/issues/3975) | [RFC]: Int8 Activation Quantization | tlrmchlsmth | 2024-04-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-3975-rfc-int8-activation-quantization.md) |
| [#4532](https://github.com/vllm-project/vllm/issues/4532) | [RFC]: Refactor FP8 kv-cache | comaniac | 2024-05-01 | ⚪ 已关闭 | 一般 |  |  |
| [#4743](https://github.com/vllm-project/vllm/issues/4743) | [RFC]: Support specifying quant_config details in the LLM or Server entrypoints | mgoin | 2024-05-10 | ⚪ 已关闭 | 一般 |  |  |
| [#8716](https://github.com/vllm-project/vllm/issues/8716) | [RFC]: quant llm from alpindale | flozi00 | 2024-09-22 | ⚪ 已关闭 | 一般 |  |  |
| [#8913](https://github.com/vllm-project/vllm/issues/8913) | [RFC]: QuantizationConfig and QuantizeMethodBase Refactor for Simplifying Kernel Integrat… | LucasWilkinson | 2024-09-27 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-8913-rfc-quantizationconfig-and-quantizemetho.md) |
| [#16220](https://github.com/vllm-project/vllm/issues/16220) | [RFC]: Changes to support attention + quant fusion | ProExpertProg | 2025-04-07 | ⚪ 已关闭 | 一般 |  |  |
| [#28538](https://github.com/vllm-project/vllm/issues/28538) | [RFC]: 4-bit KV cache quantization through Hadamard transforms | mratsim | 2025-11-12 | ⚪ 已关闭 | 一般 |  |  |
| [#30136](https://github.com/vllm-project/vllm/issues/30136) | [RFC]: Deprecate Legacy Quantization Formats | robertgshaw2-redhat | 2025-12-05 | ⚪ 已关闭 | 一般 |  |  |
| [#30359](https://github.com/vllm-project/vllm/issues/30359) | [RFC] [QeRL]: Online Quantization and Model Reloading | kylesayrs | 2025-12-09 | ⚪ 已关闭 | 一般 |  |  |
| [#30360](https://github.com/vllm-project/vllm/issues/30360) | [RFC]: Consolidate FP8 min/max values into somewhere reasonable (Python only) | rasmith | 2025-12-09 | ⚪ 已关闭 | 一般 |  |  |
| [#30663](https://github.com/vllm-project/vllm/issues/30663) | [RFC]: Consolidate Intel Quantization Toolkit Integration in vLLM | yiliu30 | 2025-12-15 | ⚪ 已关闭 | 一般 |  |  |
| [#30717](https://github.com/vllm-project/vllm/issues/30717) | [RFC]: Token Padding Strategy for FP8 GEMM Performance Optimization | 0xjunhao | 2025-12-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-30717-rfc-token-padding-strategy-for-fp8-gemm.md) |
| [#31028](https://github.com/vllm-project/vllm/issues/31028) | [RFC]: AMD-Quark Online Quantization | hangy-amd | 2025-12-19 | ⚪ 已关闭 | 一般 |  |  |
| [#32412](https://github.com/vllm-project/vllm/issues/32412) | [RFC]: online quantization user facing API | vkuzo | 2026-01-15 | ⚪ 已关闭 | 一般 |  |  |
| [#32589](https://github.com/vllm-project/vllm/issues/32589) | [RFC]: About design of QuantKey | hangy-amd | 2026-01-19 | ⚪ 已关闭 | 一般 |  |  |
| [#33301](https://github.com/vllm-project/vllm/issues/33301) | [RFC]: Support FP8 LoRA inference | jeejeelee | 2026-01-29 | ⚪ 已关闭 | 一般 |  |  |
| [#33314](https://github.com/vllm-project/vllm/issues/33314) | [RFC] [Feature]: Apply RFC #8913 to Quantized Linear Methods (Decouple Kernels from Check… | BadrBasowid | 2026-01-29 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33314-rfc-feature-apply-rfc-8913-to-quantized.md) |
| [#34331](https://github.com/vllm-project/vllm/issues/34331) | [RFC]: Ahead of time dequantization of weights for quantization emulation (OCP MX, NVFP4)… | fxmarty-amd | 2026-02-11 | ⚪ 已关闭 | 一般 |  |  |
| [#37319](https://github.com/vllm-project/vllm/issues/37319) | [RFC]: Extensible Per-Token Quantized KV Cache Scale Infrastructure | JartX | 2026-03-17 | ⚪ 已关闭 | 一般 |  |  |
| [#37908](https://github.com/vllm-project/vllm/issues/37908) | [RFC]: Add Nunchaku SVDQuant W4A4 quantization backend | ultism | 2026-03-23 | ⚪ 已关闭 | 一般 | 05007) (W4A4 with low-rank correction) is currently the onl… |  |
| [#38694](https://github.com/vllm-project/vllm/issues/38694) | [RFC]: O(1) KV Cache for vLLM: 4.8x Speedup & 22x More Accurate than TurboQuant on Qwen2.… | JEWONMOON | 2026-04-01 | ⚪ 已关闭 | 一般 | 8x faster**) |  |
| [#39583](https://github.com/vllm-project/vllm/issues/39583) | [RFC]: Migrate bitsandbytes and GGUF quantization support to OOT plugin | mgoin | 2026-04-11 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-39583-rfc-migrate-bitsandbytes-and-gguf-quanti.md) |
| [#40182](https://github.com/vllm-project/vllm/issues/40182) | [RFC]: Unified ModelOpt Quantization in vLLM | juhi10071998 | 2026-04-18 | 🟢 开放 | 一般 |  |  |
| [#40675](https://github.com/vllm-project/vllm/issues/40675) | [RFC] Support Intel Quantization Toolkit AutoRound on Intel Platforms | Zhenzhong1 | 2026-04-23 | ⚪ 已关闭 | 一般 |  |  |
| [#41629](https://github.com/vllm-project/vllm/issues/41629) | [RFC]: Route TorchAO and LLM-Compressor Quantized Inference through zentorch on AMD Zen C… | aadwived | 2026-05-04 | ⚪ 已关闭 | 一般 |  |  |
| [#42284](https://github.com/vllm-project/vllm/issues/42284) | [RFC]: support hopper FP8 MegaMoE backend for DeepSeek-V4 | RuneFang | 2026-05-11 | ⚪ 已关闭 | 一般 |  |  |
| [#44578](https://github.com/vllm-project/vllm/issues/44578) | [RFC] Add KVarN: Variance-Normalized KV-Cache Quantization (4-bit K, 2-bit V) | jakejimenez | 2026-06-04 | ⚪ 已关闭 | 一般 |  |  |
| [#46613](https://github.com/vllm-project/vllm/issues/46613) | [RFC]: KVarN: a calibration-free, variance-normalized sub-8-bit KV-cache quantization bac… | philippebich | 2026-06-24 | 🟢 开放 | 一般 | 7x capacity but give up **40 to 52% of throughput**, and ag… |  |
| [#47095](https://github.com/vllm-project/vllm/issues/47095) | [RFC]: Extension points for backends whose physical layout isn't a permutation of the log… | ChuanLi1101 | 2026-06-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47095-rfc-extension-points-for-backends-whose.md) |
| [#47684](https://github.com/vllm-project/vllm/issues/47684) | [RFC]: FlashInfer NVFP4 KV serving on pre-SM100 GPUs | lesj0610 | 2026-07-06 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47684-rfc-flashinfer-nvfp4-kv-serving-on-pre-s.md) |
| [#48374](https://github.com/vllm-project/vllm/issues/48374) | [RFC]: fp8 KV cache for the Ampere sparse-MLA path (TRITON_MLA_SPARSE) via software dequa… | nickus | 2026-07-11 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48374-rfc-fp8-kv-cache-for-the-ampere-sparse-m.md) |
| [#50281](https://github.com/vllm-project/vllm/issues/50281) | [RFC]: Per-layer online quantization configuration | fxmarty-amd | 2026-07-29 | ⚪ 已关闭 | 一般 |  |  |
| [#52202](https://github.com/vllm-project/vllm/issues/52202) | [RFC]: fp8_ds_mla KV cache on pre-SM89 (Ampere) GPUs via software-dequant TritonMLA decode | nickus | 2026-08-13 | 🟢 开放 | 一般 |  |  |
| [#53192](https://github.com/vllm-project/vllm/issues/53192) | [RFC]: V1/V2 Weight Reload with Streaming Quantization Units | new-TonyWang | 2026-08-21 | ⚪ 已关闭 | 一般 |  |  |
| [#54426](https://github.com/vllm-project/vllm/issues/54426) | [RFC] Qwen3.8-Flash-Next: fp8_e4m3 KV cache on the QSA path — working patch, one machine,… | Nanetnounou | 2026-08-30 | 🟢 开放 | 一般 |  |  |
| [#54704](https://github.com/vllm-project/vllm/issues/54704) | [RFC]: fp8 KV cache support for Inkling on SM100-family GPUs | drkm-eng | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#55196](https://github.com/vllm-project/vllm/issues/55196) | [RFC]: fp8 KV cache gives little to no memory benefit on Mamba/GDN hybrid models | shivasathishs-rp | 2026-09-03 | 🟢 开放 | 一般 |  |  |
| [#56127](https://github.com/vllm-project/vllm/issues/56127) | RFC: per-M, per-shape kernel dispatch for blockwise-FP8 linear layers (W8A16 at decode, F… | Zeuss5 | 2026-09-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-56127-rfc-per-m-per-shape-kernel-dispatch-for.md) |
| [#59016](https://github.com/vllm-project/vllm/issues/59016) | [RFC]: Software-dequant fp8 KV cache for MLA on Ampere (sm80/sm86) — consolidate the exis… | ming616 | 2026-09-28 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-59016-rfc-software-dequant-fp8-kv-cache-for-ml.md) |
| [#59062](https://github.com/vllm-project/vllm/issues/59062) | [RFC]: Online quantization from arbitrary precision | fxmarty-amd | 2026-09-28 | ⚪ 已关闭 | 一般 |  |  |
| [#64725](https://github.com/vllm-project/vllm/issues/64725) | [RFC]: Integrating LeapQuant into vLLM — INT8 Quantization of GDN/KDA Linear Attention Re… | lihaoyang-amd | 2026-10-30 | 🟢 开放 | 一般 |  |  |
| [#65560](https://github.com/vllm-project/vllm/issues/65560) | [Tracking] Replace Marlin with Humming across NVIDIA architectures | mgoin | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#65707](https://github.com/vllm-project/vllm/issues/65707) | [RFC]: [Draft] Split FusedMoEQuantConfig into static schema and runtime state | bnellnm | 2026-11-04 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-65707-rfc-draft-split-fusedmoequantconfig-into.md) |

## Disaggregated Serving(分离式部署) (53)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5557](https://github.com/vllm-project/vllm/issues/5557) | [RFC]: Implement disaggregated prefilling via KV cache transfer | KuntaiDu | 2024-06-14 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-5557-rfc-implement-disaggregated-prefilling-v.md) |
| [#10727](https://github.com/vllm-project/vllm/issues/10727) | [RFC]: Implement disaggregated prefilling using Mooncake | alogfans | 2024-11-28 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-10727-rfc-implement-disaggregated-prefilling-u.md) |
| [#13020](https://github.com/vllm-project/vllm/issues/13020) | [RFC]: Async KV Cache Transfer for Disaggregated Inference | VertexC | 2025-02-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-13020-rfc-async-kv-cache-transfer-for-disaggre.md) |
| [#19038](https://github.com/vllm-project/vllm/issues/19038) | [RFC]: Prefill-only optimizations for PD disaggregation in vLLM | KuntaiDu | 2025-06-02 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-19038-rfc-prefill-only-optimizations-for-pd-di.md) |
| [#22817](https://github.com/vllm-project/vllm/issues/22817) | [RFC]: Disaggregated Everything - Token In <> Token Out API Server | robertgshaw2-redhat | 2025-08-13 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22817-rfc-disaggregated-everything-token-in-to.md) |
| [#24256](https://github.com/vllm-project/vllm/issues/24256) | [RFC]: Add a cache hit threshold to handle Preemptions in PD-Disaggregation and enable li… | kfirwolfson | 2025-09-04 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-24256-rfc-add-a-cache-hit-threshold-to-handle.md) |
| [#24309](https://github.com/vllm-project/vllm/issues/24309) | [RFC]: Optimizing the KV cache transmission mechanism in PD disaggregation to reduce its … | CalvinXKY | 2025-09-05 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-24309-rfc-optimizing-the-kv-cache-transmission.md) |
| [#25981](https://github.com/vllm-project/vllm/issues/25981) | [RFC]: Fix for NIXL metadata exchange issue when the worker is multi-node | GuanLuo | 2025-09-30 | ⚪ 已关闭 | 一般 |  |  |
| [#26744](https://github.com/vllm-project/vllm/issues/26744) | [RFC]: Nixl Connector Heterogeneous BlockSize support | xuechendi | 2025-10-13 | ⚪ 已关闭 | 一般 |  |  |
| [#27094](https://github.com/vllm-project/vllm/issues/27094) | [RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference | MerHS | 2025-10-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-27094-rfc-remove-redundant-multi-modal-input-p.md) |
| [#27584](https://github.com/vllm-project/vllm/issues/27584) | [RFC]: Elastic Attn-FFN Disaggregation | wabluy | 2025-10-27 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-27584-rfc-elastic-attn-ffn-disaggregation.md) |
| [#31064](https://github.com/vllm-project/vllm/issues/31064) | [RFC]: Optional Hidden States Transfer in `KVConnectorBase_V1` to Eliminate Decoder Prefi… | xinyu-intel | 2025-12-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-31064-rfc-optional-hidden-states-transfer-in-k.md) |
| [#32659](https://github.com/vllm-project/vllm/issues/32659) | [RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation | fake0fan | 2026-01-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32659-rfc-tracking-follow-up-progress-on-encod.md) |
| [#34407](https://github.com/vllm-project/vllm/issues/34407) | [RFC]: Disaggregated Frontend — Separating Online Serving from Engine | hyeongyun0916 | 2026-02-12 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-34407-rfc-disaggregated-frontend-separating-on.md) |
| [#35213](https://github.com/vllm-project/vllm/issues/35213) | [RFC]: Disaggregate RendererClient from EngineClient as Foundation for Disaggregated Fron… | sagearc | 2026-02-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-35213-rfc-disaggregate-rendererclient-from-eng.md) |
| [#35780](https://github.com/vllm-project/vllm/issues/35780) | [RFC]: Remove Per-Block KV Transfer Error Handling | robertgshaw2-redhat | 2026-03-02 | ⚪ 已关闭 | 一般 |  |  |
| [#36780](https://github.com/vllm-project/vllm/issues/36780) | [RFC][NixlConnector]: Add support for hybrid SSM-FA models | NickLucche | 2026-03-11 | ⚪ 已关闭 | 一般 |  |  |
| [#36923](https://github.com/vllm-project/vllm/issues/36923) | [RFC]: [KV Connector]: Support KV push from Prefill to Decode node using Nixl Connector | snadampal | 2026-03-12 | ⚪ 已关闭 | 一般 |  |  |
| [#38474](https://github.com/vllm-project/vllm/issues/38474) | [RFC]: Add Mooncake Store Connector for Shared KV Cache Reuse | LCAIZJ | 2026-03-29 | ⚪ 已关闭 | 一般 |  |  |
| [#39766](https://github.com/vllm-project/vllm/issues/39766) | [RFC]: Support Mooncake Based ECConnector for EPD | shen-shanshan | 2026-04-14 | ⚪ 已关闭 | 一般 |  |  |
| [#42729](https://github.com/vllm-project/vllm/issues/42729) | [RFC]: Derender Endpoints to Provide Detokenization for Disaggregated Serving | hickeyma | 2026-05-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42729-rfc-derender-endpoints-to-provide-detoke.md) |
| [#42826](https://github.com/vllm-project/vllm/issues/42826) | [RFC]: Split Flashattn Forward for Prefill/Decode Separation | lianjiezh | 2026-05-16 | ⚪ 已关闭 | 一般 |  |  |
| [#43470](https://github.com/vllm-project/vllm/issues/43470) | [Feature]: [RFC] Bottleneck-aware expansion for PD disaggregation | zsn6493 | 2026-05-23 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-43470-feature-rfc-bottleneck-aware-expansion-f.md) |
| [#43807](https://github.com/vllm-project/vllm/issues/43807) | [RFC]: Deprecate `kv_both` for NIXLConnector and Enforce Explicit P/D Roles | NickLucche | 2026-05-27 | ⚪ 已关闭 | 一般 |  |  |
| [#46016](https://github.com/vllm-project/vllm/issues/46016) | [RFC]: Cross-connector KV transfer QoS — plumb request priority to the connector layer | catyans | 2026-06-18 | ⚪ 已关闭 | 一般 |  |  |
| [#46107](https://github.com/vllm-project/vllm/issues/46107) | [RFC]: Heterogeneous TP<->DP parallelism across the P/D disaggregation boundary (MoRIIO) | edwinlim0919 | 2026-06-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-46107-rfc-heterogeneous-tp-dp-parallelism-acro.md) |
| [#47161](https://github.com/vllm-project/vllm/issues/47161) | [RFC]: Streaming Derender for Disaggregated Serving | hickeyma | 2026-06-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47161-rfc-streaming-derender-for-disaggregated.md) |
| [#48117](https://github.com/vllm-project/vllm/issues/48117) | [RFC]: [ROCm][DistInf]: Distributed Inference (Disaggregated P/D + Wide-EP) CI Enablement… | lcskrishna | 2026-07-09 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48117-rfc-rocm-distinf-distributed-inference-d.md) |
| [#49765](https://github.com/vllm-project/vllm/issues/49765) | [RFC]: Native Disaggregated Pull-Based Queue Worker Interface and Heuristic Pull Router | aojea | 2026-07-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49765-rfc-native-disaggregated-pull-based-queu.md) |
| [#50913](https://github.com/vllm-project/vllm/issues/50913) | [RFC]: Cutting P/D-disaggregation TTFT: first-token seeding, fast KV side channels, and c… | dczhu | 2026-08-03 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-50913-rfc-cutting-p-d-disaggregation-ttft-firs.md) |
| [#51930](https://github.com/vllm-project/vllm/issues/51930) | [RFC]: Layerwise KV Cache Transfer for MooncakeStore with Session API Optimization | ClockOfDestiny | 2026-08-12 | 🟢 开放 | 一般 |  |  |
| [#53703](https://github.com/vllm-project/vllm/issues/53703) | [RFC]: Progressive block handoff for NIXL Push | yimdev | 2026-08-25 | 🟢 开放 | 一般 |  |  |
| [#55584](https://github.com/vllm-project/vllm/issues/55584) | [RFC]: R3 support for prefill/decode disaggregation | ShuoleiWang | 2026-09-06 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-55584-rfc-r3-support-for-prefill-decode-disagg.md) |
| [#55598](https://github.com/vllm-project/vllm/issues/55598) | [RFC]: [Tracking] Heterogeneous-TP KV sharing in Mooncake Store Connector | z-zanez | 2026-09-06 | 🟢 开放 | 一般 |  |  |
| [#57373](https://github.com/vllm-project/vllm/issues/57373) | [RFC]: Share KV transfer planning primitives across KV connectors | CAICAIIs | 2026-09-17 | 🟢 开放 | 一般 |  |  |
| [#57530](https://github.com/vllm-project/vllm/issues/57530) | [RFC]: A failure spine for the Mooncake connectors | CAICAIIs | 2026-09-18 | 🟢 开放 | 一般 |  |  |
| [#58520](https://github.com/vllm-project/vllm/issues/58520) | [RFC]: Publish Mooncake Store residency once per complete block | CAICAIIs | 2026-09-24 | 🟢 开放 | 一般 |  |  |
| [#59111](https://github.com/vllm-project/vllm/issues/59111) | [RFC]: NixlConnector: convert standard KV pages to ROCm AITER shuffled pages on the decod… | ztang2370 | 2026-09-29 | 🟢 开放 | 一般 |  |  |
| [#59141](https://github.com/vllm-project/vllm/issues/59141) | [RFC][KV Connector] NIXL support for shared in-flight prefix loads | NolenLiang | 2026-09-29 | 🟢 开放 | 一般 |  |  |
| [#60523](https://github.com/vllm-project/vllm/issues/60523) | [RFC]: Improving MooncakeConnector | wangxiyuan | 2026-10-08 | 🟢 开放 | 一般 |  |  |
| [#60986](https://github.com/vllm-project/vllm/issues/60986) | [RFC]: Scheduler-owned KV transfer obligations: release blocks only after both recv and s… | OscarSavNS | 2026-10-10 | 🟢 开放 | 一般 |  |  |
| [#61291](https://github.com/vllm-project/vllm/issues/61291) | [RFC]: use extensible token-aligned data format for aux output mooncake data | walterbm | 2026-10-12 | 🟢 开放 | 一般 |  |  |
| [#61567](https://github.com/vllm-project/vllm/issues/61567) | [RFC]: Early KV transfer for direct Mooncake P/D | NolenLiang | 2026-10-14 | 🟢 开放 | 一般 |  |  |
| [#61787](https://github.com/vllm-project/vllm/issues/61787) | [RFC]: Layerwise KV Pooling and P/D Disaggregation for High-Hit-Rate Agent Workloads | weijinqian0 | 2026-10-15 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-61787-rfc-layerwise-kv-pooling-and-p-d-disaggr.md) |
| [#62150](https://github.com/vllm-project/vllm/issues/62150) | [RFC]: PD disaggregation + RL lifecycle (pause, sleep, abort) tracking | aoshen02 | 2026-10-17 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-62150-rfc-pd-disaggregation-rl-lifecycle-pause.md) |
| [#62243](https://github.com/vllm-project/vllm/issues/62243) | [RFC]: Module layout for the Mooncake P2P KV connector | CAICAIIs | 2026-10-18 | 🟢 开放 | 一般 |  |  |
| [#63006](https://github.com/vllm-project/vllm/issues/63006) | [RFC]: Reduce descriptor and WR overhead to improve Mooncake KV transfer bandwidth | zhou-yuhan | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#64039](https://github.com/vllm-project/vllm/issues/64039) | [RFC]: Disaggregated pre/post-processing for pooling IO Processor plugins | kon-pap | 2026-10-27 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-64039-rfc-disaggregated-pre-post-processing-fo.md) |
| [#64600](https://github.com/vllm-project/vllm/issues/64600) | [RFC]: Accept already-decoded images over NIXL in the vllm-rs gRPC API | KrishnanPrash | 2026-10-29 | 🟢 开放 | 一般 |  |  |
| [#64648](https://github.com/vllm-project/vllm/issues/64648) | [RFC]: Direct layerwise and chunk-wise KV WRITE with Mooncake | nwpu-zxr | 2026-10-30 | 🟢 开放 | 一般 |  |  |
| [#65682](https://github.com/vllm-project/vllm/issues/65682) | [RFC]: Bounded load/save admission for Mooncake Store | shaoyuyoung | 2026-11-04 | ⚪ 已关闭 | 一般 |  |  |
| [#65714](https://github.com/vllm-project/vllm/issues/65714) | [RFC]: A KV compatibility identity, enforced on NIXL and P2P transfers and readable by ro… | vMaroon | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#66231](https://github.com/vllm-project/vllm/issues/66231) | [RFC]: P/D disaggregation for continuous RL rollouts (S1): order-free, seconds-long weigh… | BeichenHuang | 2026-11-07 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-66231-rfc-p-d-disaggregation-for-continuous-rl.md) |

## CUDA Graph / Compiler(CUDA Graph 与编译) (33)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#6378](https://github.com/vllm-project/vllm/issues/6378) | [RFC]: A Graph Optimization System in vLLM using torch.compile | bnellnm | 2024-07-12 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6378-rfc-a-graph-optimization-system-in-vllm.md) |
| [#6994](https://github.com/vllm-project/vllm/issues/6994) | [RFC]: More rigorous compilation warnings | tlrmchlsmth | 2024-07-31 | ⚪ 已关闭 | 一般 |  |  |
| [#16282](https://github.com/vllm-project/vllm/issues/16282) | [RFC]: How to handle the compilation of PyTorch/XLA in vLLM | yaochengji | 2025-04-08 | ⚪ 已关闭 | 一般 |  |  |
| [#16501](https://github.com/vllm-project/vllm/issues/16501) | [RFC]: vLLM x torch.compile caching should be opt-out by default | zou3519 | 2025-04-11 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-16501-rfc-vllm-x-torch-compile-caching-should.md) |
| [#20098](https://github.com/vllm-project/vllm/issues/20098) | [RFC]: Lazy CUDA Graph capture | lionelvillard | 2025-06-25 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-20098-rfc-lazy-cuda-graph-capture.md) |
| [#20283](https://github.com/vllm-project/vllm/issues/20283) | [RFC][UX][torch.compile][CUDAGraph]: Overhaul `CompilationConfig` and improve CLI `-O<n>` | ProExpertProg | 2025-06-30 | ⚪ 已关闭 | 一般 |  |  |
| [#20394](https://github.com/vllm-project/vllm/issues/20394) | [RFC][UX]: debug mode for vLLM-compile | zou3519 | 2025-07-02 | ⚪ 已关闭 | 一般 |  |  |
| [#20402](https://github.com/vllm-project/vllm/issues/20402) | [RFC]: vLLM-compile (minus cudagraphs) warm-start time should be close to zero | zou3519 | 2025-07-02 | ⚪ 已关闭 | 一般 |  |  |
| [#20451](https://github.com/vllm-project/vllm/issues/20451) | [RFC]: vLLM-compile low-hanging fruit cold start improvements | zou3519 | 2025-07-03 | ⚪ 已关闭 | 一般 |  |  |
| [#22201](https://github.com/vllm-project/vllm/issues/22201) | [RFC]: Integrate MPK (Mirage) compiler as an experimental execution backend to vLLM | ProExpertProg | 2025-08-04 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22201-rfc-integrate-mpk-mirage-compiler-as-an.md) |
| [#23261](https://github.com/vllm-project/vllm/issues/23261) | [RFC]: Address piecewise graph splitting and attention fusion incompatibility | ProExpertProg | 2025-08-20 | ⚪ 已关闭 | 一般 |  |  |
| [#27072](https://github.com/vllm-project/vllm/issues/27072) | [RFC]: Move custom fusion passes to earlier in the compilation pipeline | ProExpertProg | 2025-10-17 | ⚪ 已关闭 | 一般 |  |  |
| [#27080](https://github.com/vllm-project/vllm/issues/27080) | [RFC]: To Inductor partition or to not Inductor partition (by default in v0.11.1) | ProExpertProg | 2025-10-17 | ⚪ 已关闭 | 🔥 重点 | The startup cost is around 2-5x depending on the model - al… | [卡片](rfc-cards/rfc-27080-rfc-to-inductor-partition-or-to-not-indu.md) |
| [#27832](https://github.com/vllm-project/vllm/issues/27832) | [RFC]: Remap `CompilationConfig` from `-O` to `-cc` in CLI | ProExpertProg | 2025-10-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-27832-rfc-remap-compilationconfig-from-o-to-cc.md) |
| [#33224](https://github.com/vllm-project/vllm/issues/33224) | [RFC]: Deprecate and delete InductorAdaptor in favor of InductorStandaloneAdaptor | zou3519 | 2026-01-28 | ⚪ 已关闭 | 一般 |  |  |
| [#33645](https://github.com/vllm-project/vllm/issues/33645) | [RFC]: Improve Dynamo compile times 5-10x via unsafe assumptions | zou3519 | 2026-02-03 | ⚪ 已关闭 | 一般 | I think this proposal can make the Dynamo graph capture tim… |  |
| [#33804](https://github.com/vllm-project/vllm/issues/33804) | [RFC]: [compile] Rollout strategy for AOT Compilation. | zhxchen17 | 2026-02-04 | ⚪ 已关闭 | 一般 |  |  |
| [#34956](https://github.com/vllm-project/vllm/issues/34956) | [RFC]: Overlap weight loading with torch.compile compilation | zou3519 | 2026-02-20 | 🟢 开放 | 一般 |  |  |
| [#35771](https://github.com/vllm-project/vllm/issues/35771) | [RFC][torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation | ProExpertProg | 2026-03-02 | ⚪ 已关闭 | 一般 |  |  |
| [#38175](https://github.com/vllm-project/vllm/issues/38175) | [RFC]: Support ViT Full CUDA Graph (Tracker) | shen-shanshan | 2026-03-26 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-38175-rfc-support-vit-full-cuda-graph-tracker.md) |
| [#38744](https://github.com/vllm-project/vllm/issues/38744) | [RFC][vLLM IR]: Automatically compile native impl for IR ops | ProExpertProg | 2026-04-01 | ⚪ 已关闭 | 一般 |  |  |
| [#41296](https://github.com/vllm-project/vllm/issues/41296) | [RFC]: Custom graph dumping for vLLM-compile and depyf deprecation | ProExpertProg | 2026-04-29 | ⚪ 已关闭 | 一般 |  |  |
| [#43224](https://github.com/vllm-project/vllm/issues/43224) | [RFC]: Porting compiler fusions to manual fusion | mgoin | 2026-05-20 | 🟢 开放 | 一般 |  |  |
| [#45069](https://github.com/vllm-project/vllm/issues/45069) | [RFC]:  Profiler support for CUDA graph capture tracing and roofline trace annotations | devalshahamd | 2026-06-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45069-rfc-profiler-support-for-cuda-graph-capt.md) |
| [#45258](https://github.com/vllm-project/vllm/issues/45258) | [RFC]: FULL cudagraph support for spec-decode drafter chain steps (currently PIECEWISE-on… | rmarnold | 2026-06-11 | ⚪ 已关闭 | 一般 |  |  |
| [#48941](https://github.com/vllm-project/vllm/issues/48941) | [RFC] Recursive Tensor Collector: Fix CUDA Graph Invalidation After Weight Reload | new-TonyWang | 2026-07-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48941-rfc-recursive-tensor-collector-fix-cuda.md) |
| [#49643](https://github.com/vllm-project/vllm/issues/49643) | [RFC]: Enable full CUDA graphs for MoRIIO READ mode via async KV-load gating | edwinlim0919 | 2026-07-23 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-49643-rfc-enable-full-cuda-graphs-for-moriio-r.md) |
| [#54688](https://github.com/vllm-project/vllm/issues/54688) | [RFC] Remove torch.compile dependency from Qwen3.8-Flash-Next NVIDIA path | gcanlin | 2026-09-01 | ⚪ 已关闭 | 一般 |  |  |
| [#55599](https://github.com/vllm-project/vllm/issues/55599) | [RFC]: [Tracking] Piecewise KV loading across multiple connectors | z-zanez | 2026-09-06 | 🟢 开放 | 一般 |  |  |
| [#57175](https://github.com/vllm-project/vllm/issues/57175) | [RFC]: Retain MRV2 DBO CUDA graph replay under DP imbalance through real-token staging | algebra-MCX | 2026-09-16 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-57175-rfc-retain-mrv2-dbo-cuda-graph-replay-un.md) |
| [#57177](https://github.com/vllm-project/vllm/issues/57177) | [RFC]: Retain MRV2 DBO CUDA Graph replay under DP imbalance through real-token staging | algebra-MCX | 2026-09-16 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-57177-rfc-retain-mrv2-dbo-cuda-graph-replay-un.md) |
| [#63757](https://github.com/vllm-project/vllm/issues/63757) | [RFC]: Context-length-bounded FULL decode CUDA graphs in Model Runner V2 | eppaneamd | 2026-10-26 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-63757-rfc-context-length-bounded-full-decode-c.md) |
| [#64262](https://github.com/vllm-project/vllm/issues/64262) | [RFC] Signature-driven JIT warmup v2 for Triton kernels | LopezCastroRoberto | 2026-10-28 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-64262-rfc-signature-driven-jit-warmup-v2-for-t.md) |

## Overlap / Scheduler(重叠与调度) (28)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#3130](https://github.com/vllm-project/vllm/issues/3130) | [RFC] Upstream Chunked Prefill | rkooo567 | 2024-03-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-3130-rfc-upstream-chunked-prefill.md) |
| [#6077](https://github.com/vllm-project/vllm/issues/6077) | [RFC]: Priority Scheduling | apatke | 2024-07-02 | ⚪ 已关闭 | 一般 |  |  |
| [#6854](https://github.com/vllm-project/vllm/issues/6854) | [RFC]: Multi-Step Scheduling | SolitaryThinker | 2024-07-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6854-rfc-multi-step-scheduling.md) |
| [#7123](https://github.com/vllm-project/vllm/issues/7123) | [RFC]: Replaceable Scheduler | NadavShmayo | 2024-08-04 | ⚪ 已关闭 | 一般 |  |  |
| [#7408](https://github.com/vllm-project/vllm/issues/7408) | [RFC]: Refactor the service pipeline to overlap GPU execution and CPU operations | sleepwalker2017 | 2024-08-12 | ⚪ 已关闭 | 一般 |  |  |
| [#9098](https://github.com/vllm-project/vllm/issues/9098) | [RFC]: hide continuous batching complexity through forward context | youkaichao | 2024-10-05 | ⚪ 已关闭 | 一般 |  |  |
| [#16969](https://github.com/vllm-project/vllm/issues/16969) | [RFC]: scheduling policy optimization  in vLLM | RuixiangMa | 2025-04-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-16969-rfc-scheduling-policy-optimization-in-vl.md) |
| [#24484](https://github.com/vllm-project/vllm/issues/24484) | [RFC]: Enhancing the Pluggable Scheduler with a Workload-Aware Adaptive Policy | sidikbro | 2025-09-09 | ⚪ 已关闭 | 一般 |  |  |
| [#26433](https://github.com/vllm-project/vllm/issues/26433) | [RFC]: Batching speculation | pkuwangh | 2025-10-08 | ⚪ 已关闭 | 一般 |  |  |
| [#29406](https://github.com/vllm-project/vllm/issues/29406) | [RFC]: Add SJF Scheduling Policy | Pr0Wh1teGivee | 2025-11-25 | ⚪ 已关闭 | 一般 | In observed production traces, a single 64K-token request i… |  |
| [#29722](https://github.com/vllm-project/vllm/issues/29722) | [RFC]: Add Balance Scheduling | GDzhu01 | 2025-11-29 | ⚪ 已关闭 | 一般 |  |  |
| [#30101](https://github.com/vllm-project/vllm/issues/30101) | [RFC]: Add Buffered Response Scheduling Policy | Pr0Wh1teGivee | 2025-12-05 | ⚪ 已关闭 | 一般 |  |  |
| [#30256](https://github.com/vllm-project/vllm/issues/30256) | [RFC]: SLA-Tiered Scheduling for Latency/Throughput Optimization | ProdByBuddha | 2025-12-08 | ⚪ 已关闭 | 一般 |  |  |
| [#37168](https://github.com/vllm-project/vllm/issues/37168) | [RFC]:  Active Coordination and Two-Zone Scheduling Mechanism for KV Cache in Long-Runnin… | xinrunxue | 2026-03-16 | 🟢 开放 | 一般 |  |  |
| [#42185](https://github.com/vllm-project/vllm/issues/42185) | [RFC]: Cache-affinity-aware request ordering for the V1 scheduler | kliukovkin | 2026-05-09 | ⚪ 已关闭 | 一般 |  |  |
| [#50497](https://github.com/vllm-project/vllm/issues/50497) | [RFC]: Per-request scheduling for MLA chunked context (prefill) | LucasWilkinson | 2026-07-31 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-50497-rfc-per-request-scheduling-for-mla-chunk.md) |
| [#50738](https://github.com/vllm-project/vllm/issues/50738) | [RFC]: Dual Batch Overlap (DBO) for Model Runner V2 | specture724 | 2026-08-02 | 🟢 开放 | 一般 |  |  |
| [#51608](https://github.com/vllm-project/vllm/issues/51608) | [RFC]: Extensible Scheduler Plugin Framework for vLLM | chaunceyjiang | 2026-08-10 | 🟢 开放 | 一般 |  |  |
| [#51788](https://github.com/vllm-project/vllm/issues/51788) | [RFC]: Suffix decoding on GPU (`suffix_gpu`) — GPU-resident suffix decoding compatible wi… | zip95297 | 2026-08-11 | 🟢 开放 | 一般 | 7x faster per step even in the CPU's best case (B=256: 1 |  |
| [#52906](https://github.com/vllm-project/vllm/issues/52906) | [RFC]: Adaptive prefill token budget based on scheduling pressure | TimoSaemann | 2026-08-19 | 🟢 开放 | 一般 |  |  |
| [#55265](https://github.com/vllm-project/vllm/issues/55265) | [RFC] Length-aware batch composition for admission scheduling — experimental evidence, fa… | Bowen-mantou | 2026-09-04 | 🟢 开放 | 一般 |  |  |
| [#55524](https://github.com/vllm-project/vllm/issues/55524) | [RFC] Mamba2: exact-replay decode so that prefill, chunked prefill and decode produce ide… | NolenLiang | 2026-09-06 | 🟢 开放 | 一般 |  |  |
| [#56126](https://github.com/vllm-project/vllm/issues/56126) | RFC: enable dual-batch overlap (DBO) for dense TP models at DP=1 | Zeuss5 | 2026-09-09 | ⚪ 已关闭 | 一般 |  |  |
| [#56581](https://github.com/vllm-project/vllm/issues/56581) | [RFC]: Streaming prompt prefill for overlapping upstream generation and downstream prefill | xu-jia-ming | 2026-09-12 | 🟢 开放 | 一般 |  |  |
| [#57187](https://github.com/vllm-project/vllm/issues/57187) | [RFC]: Scheduler-Aware Multi-Tier KV Caching for vLLM with MORI-UMBP | charxwu | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#61626](https://github.com/vllm-project/vllm/issues/61626) | [RFC]:  Eager-mode Sequence Parallelism (SP) with Collective+GEMM Overlap (Async-TP) | chaojun-zhang | 2026-10-14 | 🟢 开放 | 一般 |  |  |
| [#62532](https://github.com/vllm-project/vllm/issues/62532) | [RFC]: [V1 Scheduler][KV Offload] Eliminate 1.0–1.8s Re-Queueing Stall for Completed Asyn… | cheng-hsiang-chiu | 2026-10-19 | 🟢 开放 | 一般 |  |  |
| [#65543](https://github.com/vllm-project/vllm/issues/65543) | [RFC]: Scheduler-owned persistent state for streaming RNN-T ASR (model starts as a plugin) | pdtgct | 2026-11-04 | 🟢 开放 | 一般 |  |  |

## Attention / Kernel(注意力与算子) (56)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5083](https://github.com/vllm-project/vllm/issues/5083) | [RFC]: OpenAI Triton-only backend | bringlein | 2024-05-28 | ⚪ 已关闭 | 一般 |  |  |
| [#8002](https://github.com/vllm-project/vllm/issues/8002) | [RFC]: Build `vllm-flash-attn` from source | ProExpertProg | 2024-08-29 | ⚪ 已关闭 | 一般 |  |  |
| [#12761](https://github.com/vllm-project/vllm/issues/12761) | [RFC]: Initial support for multi-model models using cross attention in V1 | bnellnm | 2025-02-05 | ⚪ 已关闭 | 一般 |  |  |
| [#13319](https://github.com/vllm-project/vllm/issues/13319) | [RFC]: Introduce a Triton-only Transformer Execution Path in vLLM | JackChuang | 2025-02-15 | ⚪ 已关闭 | 一般 |  |  |
| [#15351](https://github.com/vllm-project/vllm/issues/15351) | [Performance][RFC]: Improving  paged attention kernel's performance | wenqiny | 2025-03-23 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-15351-performance-rfc-improving-paged-attentio.md) |
| [#17419](https://github.com/vllm-project/vllm/issues/17419) | [RFC]: Kernel Library Restructure / Packaging Split (addressing long build times) | seemethere | 2025-04-29 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-17419-rfc-kernel-library-restructure-packaging.md) |
| [#18342](https://github.com/vllm-project/vllm/issues/18342) | [RFC]: hybrid dtype: float32 for weights and activation, float16 or bfloat16 for attentio… | noooop | 2025-05-19 | ⚪ 已关闭 | 一般 |  |  |
| [#20626](https://github.com/vllm-project/vllm/issues/20626) | [RFC]: Hotness-aware multi-level KV cache management to accelerate dynamic sparse attenti… | CheYulin | 2025-07-08 | ⚪ 已关闭 | 一般 |  |  |
| [#21805](https://github.com/vllm-project/vllm/issues/21805) | [RFC][Feature]: Unified Auto-Selection Mechanism for Attention Backends | vllmellm | 2025-07-29 | ⚪ 已关闭 | 一般 |  |  |
| [#22247](https://github.com/vllm-project/vllm/issues/22247) | [RFC]: Kernel Bench Docker | flozi00 | 2025-08-05 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22247-rfc-kernel-bench-docker.md) |
| [#23801](https://github.com/vllm-project/vllm/issues/23801) | [RFC]: Some questions about vllm_flash_attn | liye0626 | 2025-08-28 | ⚪ 已关闭 | 一般 |  |  |
| [#25822](https://github.com/vllm-project/vllm/issues/25822) | [RFC][Plugin]: support loading kernels from other place | ILikeIneine | 2025-09-28 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-25822-rfc-plugin-support-loading-kernels-from.md) |
| [#26133](https://github.com/vllm-project/vllm/issues/26133) | [RFC]: Support Context Parallelism with Fully Sharded KV Cache and Ring Attention | qiruiyangmeta | 2025-10-02 | ⚪ 已关闭 | 一般 |  |  |
| [#27821](https://github.com/vllm-project/vllm/issues/27821) | [RFC]: Reorganizing ViT Abstraction and Attention Selection Logic | tjtanaa | 2025-10-30 | ⚪ 已关闭 | 一般 |  |  |
| [#27822](https://github.com/vllm-project/vllm/issues/27822) | [RFC]: Reorganizing ViT Abstraction and Attention Selection Logic | tjtanaa | 2025-10-30 | ⚪ 已关闭 | 一般 |  |  |
| [#30786](https://github.com/vllm-project/vllm/issues/30786) | [RFC]: FlashMask Attention Backend for PrefixLM Models | lucianommartins | 2025-12-16 | ⚪ 已关闭 | 🔥 重点 | The 2-4x overhead applies to every attention layer, accumul… | [卡片](rfc-cards/rfc-30786-rfc-flashmask-attention-backend-for-pref.md) |
| [#31467](https://github.com/vllm-project/vllm/issues/31467) | [RFC]: A Triton operator dispatch mechanism through modified `CustomOp` | MengqingCao | 2025-12-29 | ⚪ 已关闭 | 一般 |  |  |
| [#31567](https://github.com/vllm-project/vllm/issues/31567) | [RFC]: Why custom_mask is not exposed on FlashInfer to get more flexible use case? | npuichigo | 2025-12-31 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-31567-rfc-why-custom-mask-is-not-exposed-on-fl.md) |
| [#31634](https://github.com/vllm-project/vllm/issues/31634) | [RFC]: Decouple page_size_bytes calculation in AttentionSpec for TPU/RPA Compatibility | Lumosis | 2026-01-03 | ⚪ 已关闭 | 一般 |  |  |
| [#31919](https://github.com/vllm-project/vllm/issues/31919) | [RFC]: Attention Restructuring Tracker | MatthewBonanni | 2026-01-07 | ⚪ 已关闭 | 一般 |  |  |
| [#33214](https://github.com/vllm-project/vllm/issues/33214) | [RFC]: XPU kernel migration to vllm-xpu-kernels | jikunshang | 2026-01-28 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33214-rfc-xpu-kernel-migration-to-vllm-xpu-ker.md) |
| [#33980](https://github.com/vllm-project/vllm/issues/33980) | [RFC]: Sparse attention KV cache offloading to support longer sequence length | zhangsicheng5 | 2026-02-06 | 🟢 开放 | 一般 |  |  |
| [#34444](https://github.com/vllm-project/vllm/issues/34444) | [RFC]: Decoupled Attention/FFN Parallelism | sungsooha | 2026-02-12 | ⚪ 已关闭 | 一般 |  |  |
| [#36459](https://github.com/vllm-project/vllm/issues/36459) | [RFC]: vLLM IR Out-of-Tree (OOT) Kernel Registration | wxsIcey | 2026-03-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-36459-rfc-vllm-ir-out-of-tree-oot-kernel-regis.md) |
| [#36602](https://github.com/vllm-project/vllm/issues/36602) | [RFC]: Make kernel/op and component tests device-agnostic for OOT plugins | romitjain | 2026-03-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-36602-rfc-make-kernel-op-and-component-tests-d.md) |
| [#37263](https://github.com/vllm-project/vllm/issues/37263) | [RFC]: Hotness-aware multi-level KV cache management to accelerate dynamic sparse attenti… | zengchuang-hw | 2026-03-17 | ⚪ 已关闭 | 一般 |  |  |
| [#37995](https://github.com/vllm-project/vllm/issues/37995) | [RFC]: Prefill Context Parallel for Qwen3.5 Hybrid Attention | Yancey0623 | 2026-03-24 | 🟢 开放 | 一般 |  |  |
| [#39760](https://github.com/vllm-project/vllm/issues/39760) | [RFC]: Async Failure Notification for Fault Tolerant EP Kernels | tzulingk | 2026-04-14 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-39760-rfc-async-failure-notification-for-fault.md) |
| [#40458](https://github.com/vllm-project/vllm/issues/40458) | [RFC]: Convert Triton kernels from raw pointers to block pointers | ohadeytan | 2026-04-21 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-40458-rfc-convert-triton-kernels-from-raw-poin.md) |
| [#40822](https://github.com/vllm-project/vllm/issues/40822) | [RFC]: [RL - WIP] Checkpoint -> Kernel Format transition graph | S1ro1 | 2026-04-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-40822-rfc-rl-wip-checkpoint-kernel-format-tran.md) |
| [#41535](https://github.com/vllm-project/vllm/issues/41535) | RFC: Memory-backend interface for KV cache hot paths and attention decode | ixu2486 | 2026-05-03 | ⚪ 已关闭 | 一般 |  |  |
| [#42419](https://github.com/vllm-project/vllm/issues/42419) | [RFC]: [V1][Attention] Add an experimental training-free sparse prefill attention backend… | Alicewithrabbit | 2026-05-12 | ⚪ 已关闭 | 一般 |  |  |
| [#42449](https://github.com/vllm-project/vllm/issues/42449) | [RFC]: Attention Backend Refactor | LucasWilkinson | 2026-05-12 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42449-rfc-attention-backend-refactor.md) |
| [#42545](https://github.com/vllm-project/vllm/issues/42545) | [RFC]: Tensor descriptor (TD) adoption strategy for vLLM Triton kernels | afierka-intel | 2026-05-13 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42545-rfc-tensor-descriptor-td-adoption-strate.md) |
| [#43235](https://github.com/vllm-project/vllm/issues/43235) | [RFC] NSA is architecturally incompatible with NVIDIA consumer/workstation GPUs — MLA is … | piweb-labs | 2026-05-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-43235-rfc-nsa-is-architecturally-incompatible.md) |
| [#45133](https://github.com/vllm-project/vllm/issues/45133) | [RFC]: Triton Kernel Dispatcher for Multi-Platform Support | wangxiyuan | 2026-06-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45133-rfc-triton-kernel-dispatcher-for-multi-p.md) |
| [#45872](https://github.com/vllm-project/vllm/issues/45872) | [RFC]: A registry + CI ratchet so tuned Triton kernel configs cannot silently rot | access2rohit | 2026-06-16 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45872-rfc-a-registry-ci-ratchet-so-tuned-trito.md) |
| [#45997](https://github.com/vllm-project/vllm/issues/45997) | [RFC]: Kernel-agnostic constant-stride (layer-major) KV cache layout for KV connectors | xiaguan | 2026-06-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45997-rfc-kernel-agnostic-constant-stride-laye.md) |
| [#47280](https://github.com/vllm-project/vllm/issues/47280) | [RFC]: Implement return indexer topk for sparse attention. | Ronald1995 | 2026-07-01 | ⚪ 已关闭 | 一般 |  |  |
| [#47334](https://github.com/vllm-project/vllm/issues/47334) | [RFC]: Port vLLM-built Hopper and Blackwell C++ CUDA kernels to CuTeDSL | LucasWilkinson | 2026-07-01 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47334-rfc-port-vllm-built-hopper-and-blackwell.md) |
| [#47790](https://github.com/vllm-project/vllm/issues/47790) | [RFC]: Separate Flashinfer GQA into TRTLLM and flashinfer backends. | pavanimajety | 2026-07-07 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47790-rfc-separate-flashinfer-gqa-into-trtllm.md) |
| [#47919](https://github.com/vllm-project/vllm/issues/47919) | [RFC]   guard-gated sparse MLA topology index policy | teerthsharma | 2026-07-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-47919-rfc-guard-gated-sparse-mla-topology-inde.md) |
| [#48029](https://github.com/vllm-project/vllm/issues/48029) | [RFC]: Composite Attention Backends | MatthewBonanni | 2026-07-08 | ⚪ 已关闭 | 一般 |  |  |
| [#48480](https://github.com/vllm-project/vllm/issues/48480) | [RFC]: Make Triton kernel unit tests hardware-agnostic and cover untested kernels | adobrzyn | 2026-07-13 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-48480-rfc-make-triton-kernel-unit-tests-hardwa.md) |
| [#50324](https://github.com/vllm-project/vllm/issues/50324) | [RFC]: Deprecate the FlexAttention backend | yewentao256 | 2026-07-29 | ⚪ 已关闭 | 一般 |  |  |
| [#52776](https://github.com/vllm-project/vllm/issues/52776) | [RFC]: Extensible GEMM Epilogue Contracts for Residual, Norm, and Gated Activation Fusion | calvin-fei | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#53208](https://github.com/vllm-project/vllm/issues/53208) | [RFC]: Opt-out control for individual startup kernel warmups | YongfengX | 2026-08-21 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-53208-rfc-opt-out-control-for-individual-start.md) |
| [#55394](https://github.com/vllm-project/vllm/issues/55394) | [RFC]: Tile-union QSA sparse attention for prefill (Qwen3.8-Flash-Next): share the K/V ga… | jschmied | 2026-09-04 | 🟢 开放 | 一般 |  |  |
| [#57566](https://github.com/vllm-project/vllm/issues/57566) | [RFC]: Stable, versioned plugin contract for out-of-tree attention backends + custom KV-c… | samuelsalfati | 2026-09-18 | 🟢 开放 | 一般 |  |  |
| [#58751](https://github.com/vllm-project/vllm/issues/58751) | [RFC]: Universal Triton kernel autotuning at warmup | giuseppegrossi | 2026-09-25 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-58751-rfc-universal-triton-kernel-autotuning-a.md) |
| [#60904](https://github.com/vllm-project/vllm/issues/60904) | [RFC]: A standard for Specialized (MonoKernels) Decodes in vllm/models | dllehr-amd | 2026-10-09 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-60904-rfc-a-standard-for-specialized-monokerne.md) |
| [#62083](https://github.com/vllm-project/vllm/issues/62083) | [RFC / Research Showcase] SPICY-KV: 28.6x–50x Pre-RoPE Algebraic Multigrid KV Compression… | Yuge-Y | 2026-10-16 | 🟢 开放 | 一般 |  |  |
| [#62839](https://github.com/vllm-project/vllm/issues/62839) | [RFC]: Fuse DSpark confidence-head GEMV into a single TileLang kernel | panpan0000 | 2026-10-21 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-62839-rfc-fuse-dspark-confidence-head-gemv-int.md) |
| [#63640](https://github.com/vllm-project/vllm/issues/63640) | [RFC]: Attention Backend Test Suites | tjtanaa | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#65384](https://github.com/vllm-project/vllm/issues/65384) | [RFC]: Decouple Physical PCP Topology from Per-Forward Attention Layout and Draft Executi… | pisceskkk | 2026-11-03 | 🟢 开放 | 一般 |  |  |
| [#65782](https://github.com/vllm-project/vllm/issues/65782) | [RFC]: SM90 DeepGEMM MegaMoE for DeepSeek V4 | new-TonyWang | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## KV Cache / Memory(KV Cache 与显存) (80)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#2614](https://github.com/vllm-project/vllm/issues/2614) | [RFC] Automatic Prefix Caching | zhuohan123 | 2024-01-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-2614-rfc-automatic-prefix-caching.md) |
| [#5751](https://github.com/vllm-project/vllm/issues/5751) | [RFC]: Support sparse KV cache framework | chizhang118 | 2024-06-21 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-5751-rfc-support-sparse-kv-cache-framework.md) |
| [#7697](https://github.com/vllm-project/vllm/issues/7697) | [RFC]: Enable Memory Tiering for vLLM | PanJason | 2024-08-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7697-rfc-enable-memory-tiering-for-vllm.md) |
| [#8333](https://github.com/vllm-project/vllm/issues/8333) | [RFC]: Pinned Caching with Automatic Prefix Caching (Related to Anthropic Prompt Caching … | llsj14 | 2024-09-10 | ⚪ 已关闭 | 一般 |  |  |
| [#10646](https://github.com/vllm-project/vllm/issues/10646) | [RFC]: Support KV Cache Compaction | YaoJiayi | 2024-11-25 | ⚪ 已关闭 | 一般 |  |  |
| [#10942](https://github.com/vllm-project/vllm/issues/10942) | [RFC]: Dynamic KV Cache compression based on  vLLM framework | Zachary-ai-engineer | 2024-12-06 | ⚪ 已关闭 | 🔥 重点 | 5x inference speedup for vLLM | [卡片](rfc-cards/rfc-10942-rfc-dynamic-kv-cache-compression-based-o.md) |
| [#11382](https://github.com/vllm-project/vllm/issues/11382) | [RFC]: Hybrid Memory Allocator | heheda12345 | 2024-12-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-11382-rfc-hybrid-memory-allocator.md) |
| [#12254](https://github.com/vllm-project/vllm/issues/12254) | [RFC]: Sparse KV cache management framework | ShawnD200 | 2025-01-21 | ⚪ 已关闭 | 一般 |  |  |
| [#14724](https://github.com/vllm-project/vllm/issues/14724) | [RFC]: KV Cache Offloading for Cross-Engine KV Reuse | DwyaneShi | 2025-03-13 | ⚪ 已关闭 | 一般 |  |  |
| [#15123](https://github.com/vllm-project/vllm/issues/15123) | [RFC]: layer-wise kv cache offloading to enable larger batches | sleepwalker2017 | 2025-03-19 | ⚪ 已关闭 | 一般 |  |  |
| [#16016](https://github.com/vllm-project/vllm/issues/16016) | [RFC]: Cache Salting for Secure and Flexible Prefix Caching in vLLM | dr75 | 2025-04-03 | ⚪ 已关闭 | 一般 |  |  |
| [#16144](https://github.com/vllm-project/vllm/issues/16144) | [RFC]: Offload KV cache to CPU in V1 | mengzhu28 | 2025-04-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-16144-rfc-offload-kv-cache-to-cpu-in-v1.md) |
| [#19329](https://github.com/vllm-project/vllm/issues/19329) | [RFC]: Graceful Error Handling for KV Connector Load Failures | sdavidbd | 2025-06-08 | ⚪ 已关闭 | 一般 |  |  |
| [#19854](https://github.com/vllm-project/vllm/issues/19854) | [RFC]: KV cache offloading | orozery | 2025-06-19 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-19854-rfc-kv-cache-offloading.md) |
| [#21772](https://github.com/vllm-project/vllm/issues/21772) | [RFC]: Support KV Cache sparse with common framework | FangRun2 | 2025-07-28 | ⚪ 已关闭 | 一般 |  |  |
| [#22605](https://github.com/vllm-project/vllm/issues/22605) | [RFC]: Separated CPU KV Cache Offloading/Transfer Process | ApostaC | 2025-08-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22605-rfc-separated-cpu-kv-cache-offloading-tr.md) |
| [#23641](https://github.com/vllm-project/vllm/issues/23641) | [RFC]: Frequency and Cost Aware Eviction Policy for Prefix Caching | luolun | 2025-08-26 | ⚪ 已关闭 | 一般 |  |  |
| [#25939](https://github.com/vllm-project/vllm/issues/25939) | [RFC][P/D]: Support hybrid and flexible KV Cache transfer timing & pathway at request-lev… | Shirley1Huang | 2025-09-30 | ⚪ 已关闭 | 一般 |  |  |
| [#25950](https://github.com/vllm-project/vllm/issues/25950) | [RFC]: Generalized KV cache reuse | iddo10 | 2025-09-30 | ⚪ 已关闭 | 一般 |  |  |
| [#26858](https://github.com/vllm-project/vllm/issues/26858) | [RFC]: Top-level CLI interface for KV cache offloading | ApostaC | 2025-10-15 | ⚪ 已关闭 | 一般 |  |  |
| [#27742](https://github.com/vllm-project/vllm/issues/27742) | [RFC]: KV cache layout combining all layers per block | orozery | 2025-10-29 | ⚪ 已关闭 | 一般 |  |  |
| [#27951](https://github.com/vllm-project/vllm/issues/27951) | [RFC]: Fixing the inaccurate memory profiling | zhuohan123 | 2025-11-03 | ⚪ 已关闭 | 一般 |  |  |
| [#32733](https://github.com/vllm-project/vllm/issues/32733) | [RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between… | snadampal | 2026-01-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32733-rfc-p-d-prefill-compute-optimizations-wi.md) |
| [#33398](https://github.com/vllm-project/vllm/issues/33398) | [RFC]: Layerwise KV cache offloading to support longer sequence length | zhangsicheng5 | 2026-01-30 | 🟢 开放 | 一般 |  |  |
| [#33526](https://github.com/vllm-project/vllm/issues/33526) | [RFC]: Progressive KV Cache CPU Onloading | pougetat | 2026-02-01 | ⚪ 已关闭 | 一般 |  |  |
| [#33996](https://github.com/vllm-project/vllm/issues/33996) | [RFC]: Add an option to use NCCL-based symmetric memory when pytorch symmetric memory is … | ilmarkov | 2026-02-06 | ⚪ 已关闭 | 一般 |  |  |
| [#37823](https://github.com/vllm-project/vllm/issues/37823) | [RFC] Tail-Optimized LRU (T-LRU): Reducing Tail Latency via Conversation-Aware KV Cache E… | wenxinzhang0 | 2026-03-22 | ⚪ 已关闭 | 一般 |  |  |
| [#38260](https://github.com/vllm-project/vllm/issues/38260) | [RFC]: Multi-tier KV offloading via the vLLM offloading connector | dannyharnik | 2026-03-26 | 🟢 开放 | 一般 |  |  |
| [#39305](https://github.com/vllm-project/vllm/issues/39305) | [RFC]: Selective KV Cache offload | ruocco | 2026-04-08 | ⚪ 已关闭 | 一般 |  |  |
| [#39382](https://github.com/vllm-project/vllm/issues/39382) | [RFC]: Pre-Hopper-usable 4-bit KV cache support | lesj0610 | 2026-04-09 | ⚪ 已关闭 | 一般 |  |  |
| [#39647](https://github.com/vllm-project/vllm/issues/39647) | [RFC]: support n sampling requests to eliminate redundant prefill computation and memory | Zhutianyi7230 | 2026-04-12 | 🟢 开放 | 一般 |  |  |
| [#39696](https://github.com/vllm-project/vllm/issues/39696) | [RFC]: Add optional `before_update_states` hook to `KVConnectorBase_V1` for external KV c… | naroam1 | 2026-04-13 | ⚪ 已关闭 | 一般 |  |  |
| [#43763](https://github.com/vllm-project/vllm/issues/43763) | [RFC]: Does SimpleCPUOffloadConnector have plans to support disk/ssd？ | HF-001 | 2026-05-27 | ⚪ 已关闭 | 一般 |  |  |
| [#44008](https://github.com/vllm-project/vllm/issues/44008) | [RFC]: Offloading Metrics Redesign | Srinivasoo7 | 2026-05-29 | ⚪ 已关闭 | 一般 |  |  |
| [#44223](https://github.com/vllm-project/vllm/issues/44223) | [RFC]: Semantic KV Cache Reuse Interface | zbennett10 | 2026-06-01 | 🟢 开放 | 一般 | We have observed up to 32x speedup at 16K |  |
| [#45175](https://github.com/vllm-project/vllm/issues/45175) | [RFC]: Extend hidden-state extraction to export verifier KV caches | Beichen-Ma | 2026-06-10 | ⚪ 已关闭 | 一般 |  |  |
| [#45405](https://github.com/vllm-project/vllm/issues/45405) | [RFC]: Forward ReqContext from CPUOffloadingManager to CachePolicy for session-aware KV e… | jacklin78911-collab | 2026-06-12 | 🟢 开放 | 一般 |  |  |
| [#48203](https://github.com/vllm-project/vllm/issues/48203) | [RFC]: Layerwise and Sparse KV cache offloading to support longer sequence length | zhangsicheng5 | 2026-07-10 | 🟢 开放 | 一般 |  |  |
| [#48485](https://github.com/vllm-project/vllm/issues/48485) | [RFC]: Waiting-Queue-Informed LRU for Prefix Cache Eviction | chaunceyjiang | 2026-07-13 | 🟢 开放 | 一般 |  |  |
| [#48504](https://github.com/vllm-project/vllm/issues/48504) | [RFC]: GDS support for filesystem KV-cache offloading | maheshrbapatu | 2026-07-13 | 🟢 开放 | 一般 |  |  |
| [#49413](https://github.com/vllm-project/vllm/issues/49413) | [RFC]: KV offload event path refactor — provenance-carrying events and key-only removals | Change72 | 2026-07-22 | 🟢 开放 | 一般 |  |  |
| [#50031](https://github.com/vllm-project/vllm/issues/50031) | [RFC]: Back-pressure detection for KV cache offloading tiers | bnellnm | 2026-07-27 | 🟢 开放 | 一般 |  |  |
| [#51240](https://github.com/vllm-project/vllm/issues/51240) | [RFC][kv_offload]: Tiering Admission Policy Design | orozery | 2026-08-06 | 🟢 开放 | 一般 |  |  |
| [#51428](https://github.com/vllm-project/vllm/issues/51428) | [RFC]: Programmatic Session-Aware KV Cache Management | karen-sy | 2026-08-07 | 🟢 开放 | 一般 |  |  |
| [#51439](https://github.com/vllm-project/vllm/issues/51439) | [RFC]: [TieredOffloading][Performance] Tiered Lookup Resolution | varun-sundar-rabindranath | 2026-08-07 | ⚪ 已关闭 | 一般 |  |  |
| [#51636](https://github.com/vllm-project/vllm/issues/51636) | [RFC]: Generic Control RPC for KV Connectors | liranschour | 2026-08-10 | ⚪ 已关闭 | 一般 |  |  |
| [#51639](https://github.com/vllm-project/vllm/issues/51639) | [RFC]: Generic Control RPC for KV Connectors | liranschour | 2026-08-10 | 🟢 开放 | 一般 |  |  |
| [#51751](https://github.com/vllm-project/vllm/issues/51751) | [RFC]: Pluggable KV Cache Data Types | lcfenglinwan | 2026-08-11 | ⚪ 已关闭 | 一般 |  |  |
| [#51948](https://github.com/vllm-project/vllm/issues/51948) | [RFC]: Bounded-memory video sessions — KV retention for long-running streaming-input requ… | nvbfalk | 2026-08-12 | 🟢 开放 | 一般 |  |  |
| [#52113](https://github.com/vllm-project/vllm/issues/52113) | [RFC]: Session-Aware KV Cache Hints for Agentic Workloads | FermatGo | 2026-08-13 | 🟢 开放 | 一般 |  |  |
| [#52837](https://github.com/vllm-project/vllm/issues/52837) | [RFC]: Derive KV offload save sources from prefix-cache records | Dao007forever | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#54363](https://github.com/vllm-project/vllm/issues/54363) | [RFC]: Data integrity and I/O liveness for the filesystem KV offload tier | zdtsw | 2026-08-29 | 🟢 开放 | 一般 |  |  |
| [#54777](https://github.com/vllm-project/vllm/issues/54777) | [RFC]: GDS kv offloading | effi-ofer | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#54779](https://github.com/vllm-project/vllm/issues/54779) | [RFC] Bounded, restart-safe capacity management for the filesystem KV offload tier | akalin9507 | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#55697](https://github.com/vllm-project/vllm/issues/55697) | [RFC]: Application-Directed Prefix Checkpoints for Mamba / Hybrid Prefix Caching | nicholaskh-ai | 2026-09-07 | 🟢 开放 | 一般 | 03x Kernel Speedup**: Batched phased GDN prefill kernel vs |  |
| [#57103](https://github.com/vllm-project/vllm/issues/57103) | [RFC]: Programmable KV Cache: Composable Policies for Agentic Serving | zhhangBian | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#57106](https://github.com/vllm-project/vllm/issues/57106) | [RFC]: [FS Offloading][ThreadPool] Updates to Thread Pool | varun-sundar-rabindranath | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#57228](https://github.com/vllm-project/vllm/issues/57228) | [RFC][ROCm]: Decode Context Parallelism (DCP) for DeepSeek-V4 hybrid SWA + compressed KV … | edwinlim0919 | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#57445](https://github.com/vllm-project/vllm/issues/57445) | [RFC]: Layer-Wise Mixed-Precision KV Cache | meenchen | 2026-09-17 | 🟢 开放 | 一般 |  |  |
| [#57672](https://github.com/vllm-project/vllm/issues/57672) | [RFC]: Raw-survivor storage with query-side rotation (RSQR) for KV cache eviction | null-Exception1 | 2026-09-19 | 🟢 开放 | 一般 |  |  |
| [#58562](https://github.com/vllm-project/vllm/issues/58562) | [RFC]: Pluggable KV connector metrics on the Rust frontend via connector metrics descript… | grYe99 | 2026-09-24 | 🟢 开放 | 一般 |  |  |
| [#59018](https://github.com/vllm-project/vllm/issues/59018) | [RFC]: Meta-device operator capture: see which operators a model runs on this platform, w… | sachinkademane | 2026-09-28 | 🟢 开放 | 一般 |  |  |
| [#59055](https://github.com/vllm-project/vllm/issues/59055) | [RFC]: Release all releasable GPU memory in sleep mode (tracking) | penguin-wwy | 2026-09-28 | 🟢 开放 | 一般 |  |  |
| [#59773](https://github.com/vllm-project/vllm/issues/59773) | [RFC]: Octave KV, a native 3-bit KV cache for AMD GPUs | JartX | 2026-10-02 | 🟢 开放 | 一般 |  |  |
| [#61396](https://github.com/vllm-project/vllm/issues/61396) | [RFC] [KV Offload]: Store triggers and store modes for the offloading connector (lazy off… | amirfr3 | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#62100](https://github.com/vllm-project/vllm/issues/62100) | [RFC][Experimental]: Model- and platform-customized KV cache planning | ivanium | 2026-10-16 | 🟢 开放 | 一般 |  |  |
| [#62257](https://github.com/vllm-project/vllm/issues/62257) | [RFC]: Add Single-Process In-Memory Engine Mode for Low-Tier Sandbox Environments (Colab … | ragulk143 | 2026-10-18 | 🟢 开放 | 一般 |  |  |
| [#63626](https://github.com/vllm-project/vllm/issues/63626) | [RFC]: Lossless 12-bit storage for BF16 dense weights to speed up memory-bound decode | JimmyToluene | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#63741](https://github.com/vllm-project/vllm/issues/63741) | [RFC / Performance] How LLM Agent Context Compaction Collapses Automatic Prefix Caching (… | yansu202511 | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#63744](https://github.com/vllm-project/vllm/issues/63744) | [Feature]: [RFC / Performance] How LLM Agent Context Compaction Collapses Automatic Prefi… | yansu202511 | 2026-10-26 | ⚪ 已关闭 | 一般 |  |  |
| [#64150](https://github.com/vllm-project/vllm/issues/64150) | [RFC]: Pluggable KV cache eviction policy for hybrid models | yubofredwang | 2026-10-27 | 🟢 开放 | 一般 |  |  |
| [#64175](https://github.com/vllm-project/vllm/issues/64175) | [RFC]: Serve requests on top of an externally computed KV cache (PrecomputedKVConnector) | ukanwat | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#64267](https://github.com/vllm-project/vllm/issues/64267) | [RFC]: Prefix caching of Decision 3.0 `prefix_bidir` state blocks (hit cap, in-block chec… | zamog | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#64295](https://github.com/vllm-project/vllm/issues/64295) | [RFC]: Let replicas read each other's GPU KV cache (L1 sharing) | TianDi101 | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#64401](https://github.com/vllm-project/vllm/issues/64401) | [RFC]: Large-page KV cache allocator: one KV cache group per spec | ivanium | 2026-10-29 | 🟢 开放 | 一般 |  |  |
| [#64973](https://github.com/vllm-project/vllm/issues/64973) | [RFC]: Hybrid KV Backend: Eliminating Disk Restore Pipeline Bubbles Without CPU Memory Bl… | silence-breaker | 2026-11-01 | 🟢 开放 | 一般 |  |  |
| [#65548](https://github.com/vllm-project/vllm/issues/65548) | [RFC]: Source Identity and Semantic Metadata for KV Cache Events | Chase-Rong | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#65650](https://github.com/vllm-project/vllm/issues/65650) | [RFC]: Using Backend-Owned KV Cache Runtime Shapes | Angazenn | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#65967](https://github.com/vllm-project/vllm/issues/65967) | [RFC]: Compute-free prefix KV lookup-and-pin for router-driven peer KV reuse | briefgaming | 2026-11-05 | 🟢 开放 | 一般 |  |  |
| [#65969](https://github.com/vllm-project/vllm/issues/65969) | [RFC][HiSparse] Selective shared KV offloading for DeepSeek V4.1 | Etelis | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## Parallelism / Communication(并行与通信) (19)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#18037](https://github.com/vllm-project/vllm/issues/18037) | [RFC]: Enabling Suffix Decoding, LSTM Speculator, Sequence Parallelism from Arctic Infere… | sfc-gh-aqiao | 2025-05-13 | 🟢 开放 | 🔥 重点 | - 4x faster generation with [Suffix Decoding](https://www | [卡片](rfc-cards/rfc-18037-rfc-enabling-suffix-decoding-lstm-specul.md) |
| [#22693](https://github.com/vllm-project/vllm/issues/22693) | [RFC]: Context Parallelism && Sequence Parallelism | zhenwenqi2024 | 2025-08-12 | ⚪ 已关闭 | 一般 |  |  |
| [#24657](https://github.com/vllm-project/vllm/issues/24657) | [RFC]: Support asymmetric tp/pp in p2pncclconnector | ZhongsJie | 2025-09-11 | ⚪ 已关闭 | 一般 |  |  |
| [#28912](https://github.com/vllm-project/vllm/issues/28912) | [RFC]: Support Context Pipeline Parallelism(CPP). | pisceskkk | 2025-11-18 | ⚪ 已关闭 | 一般 |  |  |
| [#30055](https://github.com/vllm-project/vllm/issues/30055) | [RFC]: Sharded Context Parallelism for Deepseek DSA | zzhx1 | 2025-12-04 | ⚪ 已关闭 | 一般 |  |  |
| [#32066](https://github.com/vllm-project/vllm/issues/32066) | [RFC]: Expose Sharding Metadata from vLLM Model Layers | chunzhao | 2026-01-09 | ⚪ 已关闭 | 一般 |  |  |
| [#33115](https://github.com/vllm-project/vllm/issues/33115) | [RFC]: Deprecate NCCL Connector? | robertgshaw2-redhat | 2026-01-26 | ⚪ 已关闭 | 一般 |  |  |
| [#33914](https://github.com/vllm-project/vllm/issues/33914) | [RFC]: Support Chunked Pipeline Parallel(CPP) with dynamic chunk size. | pisceskkk | 2026-02-05 | ⚪ 已关闭 | 一般 |  |  |
| [#40429](https://github.com/vllm-project/vllm/issues/40429) | [RFC]: Expose per-parameter sharding as DTensor metadata | junjzhang | 2026-04-21 | ⚪ 已关闭 | 一般 |  |  |
| [#41685](https://github.com/vllm-project/vllm/issues/41685) | [RFC]: Long-context-optimized Pipeline Parallelism, CPP + Async P2P + Dynamic Chunking | qiching | 2026-05-05 | ⚪ 已关闭 | 一般 |  |  |
| [#43634](https://github.com/vllm-project/vllm/issues/43634) | [RFC]:Sequence Parallelism (SP) for Vision Transformers (ViT) in VLM Inference | BossH2 | 2026-05-26 | ⚪ 已关闭 | 一般 |  |  |
| [#46439](https://github.com/vllm-project/vllm/issues/46439) | [RFC]: `nccl-m2n` for sharding-aware weight transfer in vLLM RL | kwen2501 | 2026-06-23 | 🟢 开放 | 一般 |  |  |
| [#48720](https://github.com/vllm-project/vllm/issues/48720) | [RFC] TP-wide step-consensus guard with NCCL RAS integration — field-validated, two live … | marksunner | 2026-07-15 | 🟢 开放 | 一般 |  |  |
| [#51513](https://github.com/vllm-project/vllm/issues/51513) | [RFC]: Unify functional P2P gating — one veto-only verdict for all P2P consumers (NCCL, c… | ldkhang1201 | 2026-08-08 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-51513-rfc-unify-functional-p2p-gating-one-veto.md) |
| [#53714](https://github.com/vllm-project/vllm/issues/53714) | [RFC]: PCP-Sharded O-Projection Weights | pisceskkk | 2026-08-25 | 🟢 开放 | 一般 |  |  |
| [#58329](https://github.com/vllm-project/vllm/issues/58329) | [RFC]: KVPP (KV pipeline parallel, LayerSplit) for vLLM | pisceskkk | 2026-09-23 | 🟢 开放 | 一般 |  |  |
| [#63215](https://github.com/vllm-project/vllm/issues/63215) | [RFC]: Declared ownership and layout for WeightSource (NCCL broadcast PP + NCCL M2N) | hao-aaron | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#64374](https://github.com/vllm-project/vllm/issues/64374) | [RFC]: Bounded, retryable NCCL weight updates | dharak-cohere | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#64418](https://github.com/vllm-project/vllm/issues/64418) | [RFC]: Load plan — an engine-neutral API for sharded weight transfer | hao-aaron | 2026-10-29 | 🟢 开放 | 一般 |  |  |

## Sampler / Detokenization(采样与反分词) (14)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#13361](https://github.com/vllm-project/vllm/issues/13361) | [RFC]: Deprecation of the `best_of` Sampling Parameter in vLLM V1 | WoosukKwon | 2025-02-16 | ⚪ 已关闭 | 一般 |  |  |
| [#16268](https://github.com/vllm-project/vllm/issues/16268) | [RFC]: TPU V1 Sampler planning | NickLucche | 2025-04-08 | ⚪ 已关闭 | 一般 |  |  |
| [#17191](https://github.com/vllm-project/vllm/issues/17191) | [RFC]: Custom sampling params support in REST API | afeldman-nm | 2025-04-25 | ⚪ 已关闭 | 一般 |  |  |
| [#17799](https://github.com/vllm-project/vllm/issues/17799) | [RFC]: Logits processor extensibility | afeldman-nm | 2025-05-07 | ⚪ 已关闭 | 一般 |  |  |
| [#25389](https://github.com/vllm-project/vllm/issues/25389) | [RFC]: Revise Logits Processor Programming Model | afeldman-nm | 2025-09-22 | ⚪ 已关闭 | 一般 |  |  |
| [#29081](https://github.com/vllm-project/vllm/issues/29081) | [RFC]: `SamplingParams` should raise a warning when modified via direct assignment on the… | Ki-Seki | 2025-11-20 | ⚪ 已关闭 | 一般 |  |  |
| [#31216](https://github.com/vllm-project/vllm/issues/31216) | [RFC]: Sampling Optimization: move gather of logits after argmax. | whx-sjtu | 2025-12-23 | ⚪ 已关闭 | 一般 |  |  |
| [#42259](https://github.com/vllm-project/vllm/issues/42259) | [RFC]: Logprobs/Logits Semantics and Determinism Across the vLLM Ecosystem | aoshen02 | 2026-05-11 | 🟢 开放 | 一般 |  |  |
| [#45023](https://github.com/vllm-project/vllm/issues/45023) | [RFC] Add Top-nσ logit truncation example via custom logits processor | Jackie2049 | 2026-06-09 | ⚪ 已关闭 | 一般 |  |  |
| [#46738](https://github.com/vllm-project/vllm/issues/46738) | [RFC]: Compact candidate-set renormalization for native top-k/top-p sampling | Kiid-A | 2026-06-25 | ⚪ 已关闭 | 一般 |  |  |
| [#48080](https://github.com/vllm-project/vllm/issues/48080) | [RFC]: Confident Decoding — entropy-guided intermediate-layer logits selection | zhoubiansining | 2026-07-09 | 🟢 开放 | 一般 |  |  |
| [#54143](https://github.com/vllm-project/vllm/issues/54143) | [RFC]: Backend-stable stateless RNG for Gumbel sampling | freyfwt | 2026-08-28 | 🟢 开放 | 一般 |  |  |
| [#54333](https://github.com/vllm-project/vllm/issues/54333) | [RFC]: Reduced sampling for tensor-parallel decoding | hzx55906 | 2026-08-29 | 🟢 开放 | 一般 |  |  |
| [#58959](https://github.com/vllm-project/vllm/issues/58959) | [RFC]: Return logprobs aligned with sampling-mask token IDs for score centering | aoshen02 | 2026-09-28 | ⚪ 已关闭 | 一般 |  |  |

## Frontend / Serving / API(前端与服务 API) (49)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#1968](https://github.com/vllm-project/vllm/issues/1968) | RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You… | imraviagrawal | 2023-12-07 | ⚪ 已关闭 | 一般 |  |  |
| [#6202](https://github.com/vllm-project/vllm/issues/6202) | [RFC]: Deprecate stop_reason in OpenAI Entrypoint in favor of finish_reason; fix implemen… | K-Mistele | 2024-07-08 | ⚪ 已关闭 | 一般 |  |  |
| [#6797](https://github.com/vllm-project/vllm/issues/6797) | [RFC]: Isolate OpenAI Server Into Separate Process | robertgshaw2-redhat | 2024-07-25 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6797-rfc-isolate-openai-server-into-separate.md) |
| [#8967](https://github.com/vllm-project/vllm/issues/8967) | [RFC]: Reward Modelling in OpenAI Compatible Server | noamgat | 2024-09-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-8967-rfc-reward-modelling-in-openai-compatibl.md) |
| [#9601](https://github.com/vllm-project/vllm/issues/9601) | [RFC]: openai api response format | paulcx | 2024-10-23 | ⚪ 已关闭 | 一般 |  |  |
| [#11202](https://github.com/vllm-project/vllm/issues/11202) | [RFC]: How to use python -m vllm.entrypoints.openai.api_serverto run models in my private… | cheng358 | 2024-12-14 | ⚪ 已关闭 | 一般 |  |  |
| [#11908](https://github.com/vllm-project/vllm/issues/11908) | [RFC]: Implement Structured Output support for V1 engine | russellb | 2025-01-09 | ⚪ 已关闭 | 一般 |  |  |
| [#12705](https://github.com/vllm-project/vllm/issues/12705) | [RFC]: Scale the API server across multiple CPUs | russellb | 2025-02-03 | ⚪ 已关闭 | 一般 |  |  |
| [#15681](https://github.com/vllm-project/vllm/issues/15681) | [RFC]: should we use `VLLM_WORKER_MULTIPROC_METHOD=spawn` by default for openai-compatibl… | jinzhen-lin | 2025-03-28 | ⚪ 已关闭 | 一般 |  |  |
| [#16669](https://github.com/vllm-project/vllm/issues/16669) | [RFC]: KVBlocks and Metrics Publishing In Inference Frameworks | alec-flowers | 2025-04-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-16669-rfc-kvblocks-and-metrics-publishing-in-i.md) |
| [#17076](https://github.com/vllm-project/vllm/issues/17076) | [RFC]: Implement structural_tag support in structured output | russellb | 2025-04-23 | ⚪ 已关闭 | 一般 |  |  |
| [#17817](https://github.com/vllm-project/vllm/issues/17817) | [RFC]: Unification of frontend parser | aarnphm | 2025-05-07 | ⚪ 已关闭 | 一般 |  |  |
| [#18176](https://github.com/vllm-project/vllm/issues/18176) | [RFC]: Returning Last Token, Last Layer Hidden States Through The OpenAI API | kyle-pena-kuzco | 2025-05-15 | ⚪ 已关闭 | 一般 |  |  |
| [#18420](https://github.com/vllm-project/vllm/issues/18420) | [RFC]: Support prefix and postfix addition to JSON and JSON schema for structured outputs | rishitdholakia13 | 2025-05-20 | ⚪ 已关闭 | 一般 |  |  |
| [#19097](https://github.com/vllm-project/vllm/issues/19097) | [RFC]: Response format extensions for structured outputs | aarnphm | 2025-06-03 | ⚪ 已关闭 | 一般 |  |  |
| [#22359](https://github.com/vllm-project/vllm/issues/22359) | [RFC]: How to use OpenAI's "Reasoning Levels" (low/medium/high) in vLLM? | WangRongsheng | 2025-08-06 | ⚪ 已关闭 | 一般 |  |  |
| [#25595](https://github.com/vllm-project/vllm/issues/25595) | [RFC]: Support openai HarmonyContext plugin | frank-wei | 2025-09-24 | ⚪ 已关闭 | 一般 |  |  |
| [#26110](https://github.com/vllm-project/vllm/issues/26110) | [RFC]: CI metrics dashboard improvements | rzabarazesh | 2025-10-02 | ⚪ 已关闭 | 一般 |  |  |
| [#26298](https://github.com/vllm-project/vllm/issues/26298) | [RFC]: Per-request metrics for the offline API. | maxdebayser | 2025-10-06 | ⚪ 已关闭 | 一般 |  |  |
| [#28218](https://github.com/vllm-project/vllm/issues/28218) | [RFC]: Router Registry & Plugin System for vLLM Online Serving | esmeetu | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#29632](https://github.com/vllm-project/vllm/issues/29632) | [RFC]: Force EOS generation when Structured Output Grammar is terminated | Ki-Seki | 2025-11-28 | ⚪ 已关闭 | 一般 |  |  |
| [#30696](https://github.com/vllm-project/vllm/issues/30696) | [RFC]: Per-instance EPLB metrics | markmc | 2025-12-15 | 🟢 开放 | 一般 |  |  |
| [#30778](https://github.com/vllm-project/vllm/issues/30778) | [RFC] DBO fallout tracking metrics | markmc | 2025-12-16 | ⚪ 已关闭 | 一般 |  |  |
| [#32142](https://github.com/vllm-project/vllm/issues/32142) | [RFC]: Support function calling using `structural_tag`. | chaunceyjiang | 2026-01-12 | ⚪ 已关闭 | 一般 |  |  |
| [#35004](https://github.com/vllm-project/vllm/issues/35004) | [RFC]: Realtime Endpoint Metrics | pougetat | 2026-02-21 | ⚪ 已关闭 | 一般 |  |  |
| [#37000](https://github.com/vllm-project/vllm/issues/37000) | [RFC]: add support for Cohere's /v2/embed HTTP API entry point | walterbm | 2026-03-13 | ⚪ 已关闭 | 一般 |  |  |
| [#37204](https://github.com/vllm-project/vllm/issues/37204) | [RFC]: Create separation between independent HTTP API definitions | russellb | 2026-03-16 | ⚪ 已关闭 | 一般 |  |  |
| [#38760](https://github.com/vllm-project/vllm/issues/38760) | [RFC]: Per-iteration forward pass metrics with accurate engine-level timing | tedzhouhk | 2026-04-01 | 🟢 开放 | 一般 |  |  |
| [#41893](https://github.com/vllm-project/vllm/issues/41893) | [Frontend][RFC]: Add unified Pod Snapshot API to support automatic cloud provider checkpo… | RyanRosario | 2026-05-07 | 🟢 开放 | 一般 |  |  |
| [#43015](https://github.com/vllm-project/vllm/issues/43015) | [RFC]: Add support for Cohere's /v2/chat HTTP API entry point | andrewbcohere | 2026-05-18 | ⚪ 已关闭 | 一般 |  |  |
| [#43643](https://github.com/vllm-project/vllm/issues/43643) | [RFC]: Streaming API for Real-Time Server State and Metrics | liu-cong | 2026-05-26 | ⚪ 已关闭 | 一般 |  |  |
| [#44757](https://github.com/vllm-project/vllm/issues/44757) | [Rust Frontend][RFC] Request tracing for the Rust frontend | tahsintunan | 2026-06-07 | 🟢 开放 | 一般 |  |  |
| [#47753](https://github.com/vllm-project/vllm/issues/47753) | [Rust Frontend][RFC] Anthropic Messages API: /v1/messages and /v1/messages/count_tokens | blasrodri | 2026-07-06 | 🟢 开放 | 一般 |  |  |
| [#47768](https://github.com/vllm-project/vllm/issues/47768) | [RFC]: vLLM gRPC server API for downstream integrations | connorcarpenter15 | 2026-07-06 | ⚪ 已关闭 | 一般 |  |  |
| [#47993](https://github.com/vllm-project/vllm/issues/47993) | [RFC][Frontend/CLI]: Offline prefix-cache workload analyzer | raravind007 | 2026-07-08 | 🟢 开放 | 一般 |  |  |
| [#48147](https://github.com/vllm-project/vllm/issues/48147) | [RFC]: Optimizing Asynchronous Stream Serialization in OpenAI API Entrypoint | shujaSN | 2026-07-09 | 🟢 开放 | 一般 |  |  |
| [#49411](https://github.com/vllm-project/vllm/issues/49411) | [RFC] [Feature] [Experimental] Support GigaToken Accelerated Tokenizer Mode | Prasannajaga | 2026-07-22 | 🟢 开放 | 一般 |  |  |
| [#49705](https://github.com/vllm-project/vllm/issues/49705) | [RFC]: Support router-driven mixtures of multiple LoRA adapters | navyanshmahla | 2026-07-24 | 🟢 开放 | 一般 |  |  |
| [#51186](https://github.com/vllm-project/vllm/issues/51186) | [RFC]: Adopt otel genai semantic conventions for metrics | kahowell | 2026-08-05 | 🟢 开放 | 一般 |  |  |
| [#51220](https://github.com/vllm-project/vllm/issues/51220) | [RFC][Rust Frontend] Complete RL control-plane endpoint parity | aoshen02 | 2026-08-06 | 🟢 开放 | 一般 |  |  |
| [#53452](https://github.com/vllm-project/vllm/issues/53452) | [Rust Frontend][RFC] Add Qwen3-ASR WebSocket `/v1/realtime` | TanNgocDo | 2026-08-23 | 🟢 开放 | 一般 |  |  |
| [#53484](https://github.com/vllm-project/vllm/issues/53484) | [RFC]: Generalize Connector Metrics | Srinivasoo7 | 2026-08-23 | 🟢 开放 | 一般 |  |  |
| [#53486](https://github.com/vllm-project/vllm/issues/53486) | [RFC]: Secondary Tier Usage Metrics | Srinivasoo7 | 2026-08-23 | 🟢 开放 | 一般 |  |  |
| [#56993](https://github.com/vllm-project/vllm/issues/56993) | [RFC]: Split scale-out into owned components (Renderer / Frontend / Generation) with dedi… | hickeyma | 2026-09-15 | 🟢 开放 | 一般 |  |  |
| [#57479](https://github.com/vllm-project/vllm/issues/57479) | [RFC]: Bound frontend drain latency for bulk aborts and long non-streaming completions | aoshen02 | 2026-09-18 | 🟢 开放 | 一般 |  |  |
| [#57499](https://github.com/vllm-project/vllm/issues/57499) | [RFC]: Stateless Responses API in the Rust frontend | ai-jz | 2026-09-18 | 🟢 开放 | 一般 |  |  |
| [#61423](https://github.com/vllm-project/vllm/issues/61423) | [RFC]: Re-evaluate per-iteration metrics as a foundation for orchestrator observability | markmc | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#63140](https://github.com/vllm-project/vllm/issues/63140) | [RFC] Stream request-level prefill progress over gRPC | KaisennHu | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#65797](https://github.com/vllm-project/vllm/issues/65797) | [RFC]: Token-id parity with the Python frontend as a tested contract for the Rust frontend | waynehacking8 | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## Architecture / Engine(架构与引擎) (51)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#5552](https://github.com/vllm-project/vllm/issues/5552) | [RFC]: Refactor Worker and ModelRunner to consolidate control plane communication | stephanie-wang | 2024-06-14 | ⚪ 已关闭 | 一般 |  |  |
| [#5775](https://github.com/vllm-project/vllm/issues/5775) | [RFC]: A Flexible Architecture for Distributed Inference | youkaichao | 2024-06-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-5775-rfc-a-flexible-architecture-for-distribu.md) |
| [#6556](https://github.com/vllm-project/vllm/issues/6556) | [RFC]: Single Program Multiple Data (SPMD) Worker Control Plane | ruisearch42 | 2024-07-19 | ⚪ 已关闭 | 一般 |  |  |
| [#7045](https://github.com/vllm-project/vllm/issues/7045) | [RFC]: Deprecation and removal for `--engine-use-ray` | simon-mo | 2024-08-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7045-rfc-deprecation-and-removal-for-engine-u.md) |
| [#7124](https://github.com/vllm-project/vllm/issues/7124) | [RFC]: Model architecture plugins | NadavShmayo | 2024-08-04 | ⚪ 已关闭 | 一般 |  |  |
| [#8453](https://github.com/vllm-project/vllm/issues/8453) | [RFC]: Support encode only models by Workflow Defined Engine | noooop | 2024-09-13 | ⚪ 已关闭 | 一般 |  |  |
| [#8779](https://github.com/vllm-project/vllm/issues/8779) | vLLM's V1 Engine Architecture | simon-mo | 2024-09-24 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-8779-vllm-s-v1-engine-architecture.md) |
| [#11152](https://github.com/vllm-project/vllm/issues/11152) | [RFC][Exploratory]: vLLM Neuron Backend with V1 Architecture | liangfu | 2024-12-12 | ⚪ 已关闭 | 一般 |  |  |
| [#11399](https://github.com/vllm-project/vllm/issues/11399) | [RFC]: Flexible Weight Sync for vLLM Workers | ZSL98 | 2024-12-21 | ⚪ 已关闭 | 一般 |  |  |
| [#11522](https://github.com/vllm-project/vllm/issues/11522) | [RFC]: Refactor tool parsers to eliminate coding errors and allow more efficient implemen… | elementary-particle | 2024-12-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-11522-rfc-refactor-tool-parsers-to-eliminate-c.md) |
| [#11945](https://github.com/vllm-project/vllm/issues/11945) | [RFC]: Pipeline-Parallelism for vLLM V1 | ruisearch42 | 2025-01-10 | ⚪ 已关闭 | 一般 |  |  |
| [#12363](https://github.com/vllm-project/vllm/issues/12363) | [RFC]: Refactor `config-format` and `load-format` as plugins | maxdebayser | 2025-01-23 | ⚪ 已关闭 | 一般 |  |  |
| [#12480](https://github.com/vllm-project/vllm/issues/12480) | [RFC]: [V1] TPU support and multiple architecture support | alexm-redhat | 2025-01-27 | ⚪ 已关闭 | 一般 |  |  |
| [#12992](https://github.com/vllm-project/vllm/issues/12992) | [RFC]: Device-agnostic Abstraction for V1 Architecture | liangfu | 2025-02-09 | ⚪ 已关闭 | 一般 |  |  |
| [#17140](https://github.com/vllm-project/vllm/issues/17140) | [RFC]: Native support for Mamba, SSM, and hybrid transformer models in vLLM V1 | tlrmchlsmth | 2025-04-24 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-17140-rfc-native-support-for-mamba-ssm-and-hyb.md) |
| [#18009](https://github.com/vllm-project/vllm/issues/18009) | [RFC]: Model Parallelism with Single Worker using SPMD | lsy323 | 2025-05-12 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18009-rfc-model-parallelism-with-single-worker.md) |
| [#18052](https://github.com/vllm-project/vllm/issues/18052) | [RFC]: Support pooling in V1 | 22quinn | 2025-05-13 | ⚪ 已关闭 | 一般 |  |  |
| [#18953](https://github.com/vllm-project/vllm/issues/18953) | [RFC]: vLLM configuration  refactoring and modularization | jeejeelee | 2025-05-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18953-rfc-vllm-configuration-refactoring-and-m.md) |
| [#19161](https://github.com/vllm-project/vllm/issues/19161) | [RFC]: Enhancing vLLM Plugin Architecture | kzawora-intel | 2025-06-04 | ⚪ 已关闭 | 一般 |  |  |
| [#20089](https://github.com/vllm-project/vllm/issues/20089) | [RFC]: Re-evaluating `auto` Thread Binding in `VLLM_CPU_OMP_THREADS_BIND` for SMT Archite… | Akashcodes732 | 2025-06-25 | ⚪ 已关闭 | 一般 | For instance, on a 4-socket POWER10 system with 384 logical… |  |
| [#20799](https://github.com/vllm-project/vllm/issues/20799) | [RFC]: Prototype Separating Vision Encoder to Its Own Worker | fake0fan | 2025-07-11 | ⚪ 已关闭 | 一般 |  |  |
| [#21082](https://github.com/vllm-project/vllm/issues/21082) | [RFC]: Neuron Support for V1 Engine | aarondou | 2025-07-16 | ⚪ 已关闭 | 一般 |  |  |
| [#22124](https://github.com/vllm-project/vllm/issues/22124) | [RFC]: Prompt Embeddings Support in v1 Engine | qthequartermasterman | 2025-08-02 | ⚪ 已关闭 | 一般 |  |  |
| [#22992](https://github.com/vllm-project/vllm/issues/22992) | [RFC]: Refactor CI/CD | robertgshaw2-redhat | 2025-08-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22992-rfc-refactor-ci-cd.md) |
| [#23161](https://github.com/vllm-project/vllm/issues/23161) | [RFC]: Support Layer-Specific KV-Cache Dtype & Layout in V1 Engine for Hybrid Models | zhiyuan1i | 2025-08-19 | 🟢 开放 | 一般 |  |  |
| [#23233](https://github.com/vllm-project/vllm/issues/23233) | [RFC]: Restructure the core loop to allow more asynchrony | WoosukKwon | 2025-08-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23233-rfc-restructure-the-core-loop-to-allow-m.md) |
| [#23881](https://github.com/vllm-project/vllm/issues/23881) | [RFC]: Optimize grammar_bitmask Overhead by Moving Logic to Worker_runner | GGKOP | 2025-08-29 | ⚪ 已关闭 | 一般 |  |  |
| [#32028](https://github.com/vllm-project/vllm/issues/32028) | [RFC]: EPLB Implementation Refactoring | ilmarkov | 2026-01-09 | ⚪ 已关闭 | 一般 |  |  |
| [#35409](https://github.com/vllm-project/vllm/issues/35409) | [RFC]: Worker Controller for warm GPU worker pooling to reduce cold-start latency | tangkenyi2001 | 2026-02-26 | ⚪ 已关闭 | 一般 |  |  |
| [#35858](https://github.com/vllm-project/vllm/issues/35858) | [RFC]: Unify engine process monitoring in engine manager and add Ray backend support | fangyuchu | 2026-03-03 | ⚪ 已关闭 | 一般 |  |  |
| [#39678](https://github.com/vllm-project/vllm/issues/39678) | [RFC]: Async parallel startup for EngineCore processes in DP/TP scenarios | hwhaokun | 2026-04-13 | ⚪ 已关闭 | 一般 |  |  |
| [#40953](https://github.com/vllm-project/vllm/issues/40953) | [RFC]: Refactor PassManager infrastructure | wxsIcey | 2026-04-27 | ⚪ 已关闭 | 一般 |  |  |
| [#48297](https://github.com/vllm-project/vllm/issues/48297) | [RFC] LoRA Correctness and Lifecycle for RL Training | aoshen02 | 2026-07-11 | 🟢 开放 | 一般 |  |  |
| [#48311](https://github.com/vllm-project/vllm/issues/48311) | [RFC] Engine Sleep/Wake/Drain Lifecycle for RL | aoshen02 | 2026-07-11 | 🟢 开放 | 一般 |  |  |
| [#48920](https://github.com/vllm-project/vllm/issues/48920) | [RFC]: Unify the Weight Loading Lifecycle Across Initial Load and Reload | aoshen02 | 2026-07-17 | ⚪ 已关闭 | 一般 |  |  |
| [#49288](https://github.com/vllm-project/vllm/issues/49288) | [RFC]: Multimodal Asynchronous Collaborative Architecture Sidecar | xiaonaiping114 | 2026-07-21 | 🟢 开放 | 一般 |  |  |
| [#51472](https://github.com/vllm-project/vllm/issues/51472) | [RFC] Multimodal RL inputs for /inference/v1/generate | aoshen02 | 2026-08-08 | 🟢 开放 | 一般 |  |  |
| [#52125](https://github.com/vllm-project/vllm/issues/52125) | [RFC]: vLLM Native Reusable Engine Snapshot Sleep (L3) | fuhefei | 2026-08-13 | 🟢 开放 | 一般 |  |  |
| [#54229](https://github.com/vllm-project/vllm/issues/54229) | [RFC]: Privacy-safe runtime incident snapshots for fatal EngineCore failures | jackLei0901 | 2026-08-28 | 🟢 开放 | 一般 |  |  |
| [#56816](https://github.com/vllm-project/vllm/issues/56816) | [RFC]: Add Watchdog for Engine and Worker Monitoring | wenjinhust | 2026-09-14 | ⚪ 已关闭 | 一般 |  |  |
| [#56851](https://github.com/vllm-project/vllm/issues/56851) | [RFC]: Request level text and derender output on `/inference/v1/generate` | hickeyma | 2026-09-14 | ⚪ 已关闭 | 一般 |  |  |
| [#57895](https://github.com/vllm-project/vllm/issues/57895) | [RFC]: Stable runtime tensor lifecycle for sleep and live weight reload | Ronald1995 | 2026-09-21 | 🟢 开放 | 一般 |  |  |
| [#58647](https://github.com/vllm-project/vllm/issues/58647) | [RFC]: Hardening /v1/messages (Anthropic API) for Claude Code across model families | robertgshaw2-redhat | 2026-09-25 | 🟢 开放 | 一般 |  |  |
| [#59365](https://github.com/vllm-project/vllm/issues/59365) | [RFC]: Modular read strategies and custom parameters for /v1/systemone and /v1/decisions | Timiku | 2026-09-30 | 🟢 开放 | 一般 |  |  |
| [#59502](https://github.com/vllm-project/vllm/issues/59502) | [RFC]: Weight Reload Refactor | hao-aaron | 2026-09-30 | ⚪ 已关闭 | 一般 |  |  |
| [#61880](https://github.com/vllm-project/vllm/issues/61880) | [RFC]: Native in-engine parallel beam search | bing-xbx | 2026-10-15 | 🟢 开放 | 一般 |  |  |
| [#63363](https://github.com/vllm-project/vllm/issues/63363) | [RFC]: Keep per-request token lists out of EngineCore garbage collection | gcanlin | 2026-10-23 | ⚪ 已关闭 | 一般 |  |  |
| [#64237](https://github.com/vllm-project/vllm/issues/64237) | [RFC]: KV tier on a distributed filesystem — multi-engine sharing with a cluster-wide quo… | kdtmac | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#65545](https://github.com/vllm-project/vllm/issues/65545) | [RFC]: Lifecycle hooks for endpoint plugins (readiness and graceful drain), with an out-o… | pdtgct | 2026-11-04 | 🟢 开放 | 一般 |  |  |
| [#65959](https://github.com/vllm-project/vllm/issues/65959) | [RFC]: Switch Rust regex engine for response parsing | Rocketknight1 | 2026-11-05 | 🟢 开放 | 一般 |  |  |
| [#66134](https://github.com/vllm-project/vllm/issues/66134) | [RFC]: Text sessions on /v1/realtime with streaming-input prefill | AshutoshRautela | 2026-11-06 | 🟢 开放 | 一般 |  |  |

## Hardware / Backend(硬件与后端) (27)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#7247](https://github.com/vllm-project/vllm/issues/7247) | [RFC]: Initial support for RBLN NPU | rebel-jonghewk | 2024-08-07 | ⚪ 已关闭 | 一般 |  |  |
| [#7692](https://github.com/vllm-project/vllm/issues/7692) | [RFC]: Add Ascend NPU as a new backend | wangshuai09 | 2024-08-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7692-rfc-add-ascend-npu-as-a-new-backend.md) |
| [#9268](https://github.com/vllm-project/vllm/issues/9268) | [RFC]: Make device agnostic for diverse hardware support | wangshuai09 | 2024-10-11 | ⚪ 已关闭 | 一般 |  |  |
| [#11162](https://github.com/vllm-project/vllm/issues/11162) | [RFC]: Hardware pluggable | wangxiyuan | 2024-12-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-11162-rfc-hardware-pluggable.md) |
| [#21504](https://github.com/vllm-project/vllm/issues/21504) | [RFC] [ROCm] [AITER]: Propose a `_aiter_ops` class like `_custom_ops` and `_ipex_ops` | tjtanaa | 2025-07-24 | ⚪ 已关闭 | 一般 |  |  |
| [#23786](https://github.com/vllm-project/vllm/issues/23786) | [RFC]: Design a new Layer-Pluggable abstraction to work together with CustomOp | whx-sjtu | 2025-08-28 | ⚪ 已关闭 | 一般 |  |  |
| [#24293](https://github.com/vllm-project/vllm/issues/24293) | [RFC]: Environment variable to switch backend between CPU and GPU | Sundaresan-G | 2025-09-05 | ⚪ 已关闭 | 一般 |  |  |
| [#25590](https://github.com/vllm-project/vllm/issues/25590) | [RFC]: llama.cpp CPU Backend for Enhanced CPU Performance & CI Testing | sarosh-quraishi | 2025-09-24 | ⚪ 已关闭 | 一般 |  |  |
| [#28797](https://github.com/vllm-project/vllm/issues/28797) | [RFC]: [CPU]covt_e4m3_bf16 | wangyxbh | 2025-11-16 | ⚪ 已关闭 | 一般 |  |  |
| [#30112](https://github.com/vllm-project/vllm/issues/30112) | [RFC]: Elastic DP Rank Recovery: Graceful Handling of GPU Hardware Failures Without Full … | tzulingk | 2025-12-05 | ⚪ 已关闭 | 一般 |  |  |
| [#30679](https://github.com/vllm-project/vllm/issues/30679) | [RFC]: Replace `torch.cuda` API with `torch.accelerator` for better hardware compatiblity. | jikunshang | 2025-12-15 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-30679-rfc-replace-torch-cuda-api-with-torch-ac.md) |
| [#30985](https://github.com/vllm-project/vllm/issues/30985) | [RFC]: DRY Dependencies for Better Hardware-Specific vLLM Dependency Management | wjhrdy | 2025-12-18 | ⚪ 已关闭 | 一般 | py (currently causes 3x build overhead) |  |
| [#31671](https://github.com/vllm-project/vllm/issues/31671) | [RFC]: Integrate omni-ai-npu/omni-infer's Topology-Aware Placement and Execution Optimiza… | taostronger | 2026-01-04 | 🟢 开放 | 一般 |  |  |
| [#35089](https://github.com/vllm-project/vllm/issues/35089) | [RFC]: In-Tree AMD Zen CPU Backend via zentorch | amd-lalithnc | 2026-02-23 | ⚪ 已关闭 | 一般 |  |  |
| [#37305](https://github.com/vllm-project/vllm/issues/37305) | [RFC][XPU]: Enable Intel XPU CI for vLLM | wendyliu235 | 2026-03-17 | ⚪ 已关闭 | 一般 |  |  |
| [#38974](https://github.com/vllm-project/vllm/issues/38974) | [RFC]: Request to clarify attribution/authorship for the RISC-V CPU backend PR chain (#20… | ihb2032 | 2026-04-04 | ⚪ 已关闭 | 一般 |  |  |
| [#44219](https://github.com/vllm-project/vllm/issues/44219) | [RFC]: hardware agnostic model definitions in vLLM | zou3519 | 2026-06-01 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-44219-rfc-hardware-agnostic-model-definitions.md) |
| [#44421](https://github.com/vllm-project/vllm/issues/44421) | [RFC]: AMD Zen CPU CI Infrastructure for vLLM | Chinmay-Kulkarni-AMD | 2026-06-03 | ⚪ 已关闭 | 一般 |  |  |
| [#50966](https://github.com/vllm-project/vllm/issues/50966) | [RFC]: Pluggable structured-output backends + GRID as a third contract point (declared ou… | arjmandi | 2026-08-04 | 🟢 开放 | 一般 |  |  |
| [#51212](https://github.com/vllm-project/vllm/issues/51212) | [RFC]: Model Runner V2 Pluggable Design | wangxiyuan | 2026-08-06 | 🟢 开放 | 一般 |  |  |
| [#52911](https://github.com/vllm-project/vllm/issues/52911) | [RFC]: DeepSeek-V4 Performance Optimization on ROCm (Phase Two) | shen-shanshan | 2026-08-19 | 🟢 开放 | 一般 |  |  |
| [#54766](https://github.com/vllm-project/vllm/issues/54766) | [RFC]: Custom all-reduce for XPU | recutita | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#56506](https://github.com/vllm-project/vllm/issues/56506) | [RFC]: DeepSeek-V4.1-Flash performance on ROCm | Fangzhou-Ai | 2026-09-11 | ⚪ 已关闭 | 🔥 重点 | 7x per-token speedup: the agentic trace is time-driven, so … | [卡片](rfc-cards/rfc-56506-rfc-deepseek-v4-1-flash-performance-on-r.md) |
| [#61305](https://github.com/vllm-project/vllm/issues/61305) | [RFC]: Simulated execution with pluggable timing models (building on #47922) | lucifer1004 | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#61406](https://github.com/vllm-project/vllm/issues/61406) | [RFC]: vLLM Simulator: CPU-Based CI and Hardware-Aware Performance Simulation | zhuzi-z | 2026-10-13 | 🟢 开放 | 一般 |  |  |
| [#63439](https://github.com/vllm-project/vllm/issues/63439) | [RFC]: Reduce CPU overhead in Model Runner V2 for long-context P/D serving | gcanlin | 2026-10-24 | 🟢 开放 | 一般 |  |  |
| [#63818](https://github.com/vllm-project/vllm/issues/63818) | [RFC]: Pluggable pipeline-parallel transport, and a path to out-of-process (elastic) PP s… | yanyinglin | 2026-10-26 | 🟢 开放 | 一般 |  |  |

## Feature / Ecosystem(功能与生态) (81)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#1866](https://github.com/vllm-project/vllm/issues/1866) | [RFC] Initial Support for AWS Inferentia | liangfu | 2023-11-30 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-1866-rfc-initial-support-for-aws-inferentia.md) |
| [#3620](https://github.com/vllm-project/vllm/issues/3620) | [RFC] Initial Support for Cloud TPUs | WoosukKwon | 2024-03-25 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-3620-rfc-initial-support-for-cloud-tpus.md) |
| [#3654](https://github.com/vllm-project/vllm/issues/3654) | [RFC] Initial Support for CPUs | bigPYJ1151 | 2024-03-27 | ⚪ 已关闭 | 一般 |  |  |
| [#3725](https://github.com/vllm-project/vllm/issues/3725) | [RFC] Initial support for Intel GPU | jikunshang | 2024-03-29 | ⚪ 已关闭 | 一般 |  |  |
| [#3780](https://github.com/vllm-project/vllm/issues/3780) | [RFC] How do we test and support third-party models | youkaichao | 2024-04-01 | ⚪ 已关闭 | 一般 |  |  |
| [#4194](https://github.com/vllm-project/vllm/issues/4194) | [RFC]: Multi-modality Support on vLLM | ywang96 | 2024-04-19 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-4194-rfc-multi-modality-support-on-vllm.md) |
| [#4461](https://github.com/vllm-project/vllm/issues/4461) | [RFC]: Initial support for Pipeline Paralleism | andoorve | 2024-04-29 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-4461-rfc-initial-support-for-pipeline-paralle.md) |
| [#4873](https://github.com/vllm-project/vllm/issues/4873) | [RFC]: Add control panel support for vLLM | leiwen83 | 2024-05-17 | 🟢 开放 | 一般 |  |  |
| [#5723](https://github.com/vllm-project/vllm/issues/5723) | [RFC]: Add runtime weight update API | lyuqin-scale | 2024-06-20 | ⚪ 已关闭 | 一般 |  |  |
| [#6226](https://github.com/vllm-project/vllm/issues/6226) | [RFC] Drop beam search support | WoosukKwon | 2024-07-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6226-rfc-drop-beam-search-support.md) |
| [#7480](https://github.com/vllm-project/vllm/issues/7480) | [RFC]: Add a FastChat like routing server to provide a central endpoint for multiple mode… | fozziethebeat | 2024-08-13 | ⚪ 已关闭 | 一般 |  |  |
| [#7558](https://github.com/vllm-project/vllm/issues/7558) | [RFC]: Support for video input | TKONIY | 2024-08-15 | ⚪ 已关闭 | 一般 |  |  |
| [#7769](https://github.com/vllm-project/vllm/issues/7769) | [RFC]: Keep a Changelog & Add FAQs in the Documentation | K-Mistele | 2024-08-22 | ⚪ 已关闭 | 一般 |  |  |
| [#9192](https://github.com/vllm-project/vllm/issues/9192) | [RFC]: Adopt mergify for auto-labeling PRs | russellb | 2024-10-09 | ⚪ 已关闭 | 一般 |  |  |
| [#9435](https://github.com/vllm-project/vllm/issues/9435) | [RFC]: Enable stale bot for issues and PRs | russellb | 2024-10-16 | ⚪ 已关闭 | 一般 |  |  |
| [#9652](https://github.com/vllm-project/vllm/issues/9652) | [RFC]: Add support for IBM Spyre accelerator | tdoublep | 2024-10-24 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-9652-rfc-add-support-for-ibm-spyre-accelerato.md) |
| [#11065](https://github.com/vllm-project/vllm/issues/11065) | [RFC]: Adding support for Geospatial models | christian-pinto | 2024-12-10 | ⚪ 已关闭 | 一般 |  |  |
| [#11137](https://github.com/vllm-project/vllm/issues/11137) | [RFC]: Improve Ray Support in vLLM for Enhanced Elasticity and Performance | Jeffwan | 2024-12-12 | ⚪ 已关闭 | 一般 |  |  |
| [#11410](https://github.com/vllm-project/vllm/issues/11410) | [RFC]: The two features i wish vllm has | MohamedAliRashad | 2024-12-22 | ⚪ 已关闭 | 一般 |  |  |
| [#13981](https://github.com/vllm-project/vllm/issues/13981) | [RFC]: Drop support for prompt adapter | WoosukKwon | 2025-02-27 | ⚪ 已关闭 | 一般 |  |  |
| [#14374](https://github.com/vllm-project/vllm/issues/14374) | [RFC]: Drop Support for OpenVINO | simon-mo | 2025-03-06 | ⚪ 已关闭 | 一般 |  |  |
| [#14981](https://github.com/vllm-project/vllm/issues/14981) | [RFC]: vLLM Windows CUDA support | SystemPanic | 2025-03-17 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-14981-rfc-vllm-windows-cuda-support.md) |
| [#15254](https://github.com/vllm-project/vllm/issues/15254) | [RFC]: Better support for weight updating while waking up from sleep mode for RLHF | erictang000 | 2025-03-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-15254-rfc-better-support-for-weight-updating-w.md) |
| [#16052](https://github.com/vllm-project/vllm/issues/16052) | [RFC]: Extending VLLM towards native support of non text-generating models | christian-pinto | 2025-04-04 | ⚪ 已关闭 | 一般 |  |  |
| [#17823](https://github.com/vllm-project/vllm/issues/17823) | [RFC]: Add automated profiling sweep and heatmap visualization tools | ConstBob | 2025-05-08 | ⚪ 已关闭 | 一般 |  |  |
| [#18153](https://github.com/vllm-project/vllm/issues/18153) | [RFC]: Blackwell Enablement for vLLM (SM100) | pavanimajety | 2025-05-14 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18153-rfc-blackwell-enablement-for-vllm-sm100.md) |
| [#19061](https://github.com/vllm-project/vllm/issues/19061) | [RFC]: Drop CUDA 11.8 Support | simon-mo | 2025-06-03 | ⚪ 已关闭 | 一般 |  |  |
| [#19407](https://github.com/vllm-project/vllm/issues/19407) | [RFC]: Support automatic max context length via `--max-model-len auto` | yeqcharlotte | 2025-06-10 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-19407-rfc-support-automatic-max-context-length.md) |
| [#20234](https://github.com/vllm-project/vllm/issues/20234) | [RFC]: Support Prithvi geospatial model in serving mode | mgazz | 2025-06-30 | ⚪ 已关闭 | 一般 |  |  |
| [#20298](https://github.com/vllm-project/vllm/issues/20298) | [RFC][PP]: Integrate Token Throttling into vLLM | gty111 | 2025-07-01 | ⚪ 已关闭 | 一般 |  |  |
| [#23157](https://github.com/vllm-project/vllm/issues/23157) | [RFC]: Integrate MetaX GPU Backend into vLLM Plugin | ILikeIneine | 2025-08-19 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23157-rfc-integrate-metax-gpu-backend-into-vll.md) |
| [#24163](https://github.com/vllm-project/vllm/issues/24163) | [RFC]: Support fast inplace model update by shared IPC buffer | weixiao-huang | 2025-09-03 | ⚪ 已关闭 | 一般 |  |  |
| [#24284](https://github.com/vllm-project/vllm/issues/24284) | [RFC]: Support reporting tool output tokens in OutputTokensDetails | yeqcharlotte | 2025-09-05 | ⚪ 已关闭 | 一般 |  |  |
| [#24288](https://github.com/vllm-project/vllm/issues/24288) | [RFC]: Support Returning Prompt Hidden States | charlotte12l | 2025-09-05 | ⚪ 已关闭 | 一般 |  |  |
| [#25314](https://github.com/vllm-project/vllm/issues/25314) | [Draft][RFC]: KVCache support nonuniform page size | zzzzwwjj | 2025-09-20 | ⚪ 已关闭 | 一般 |  |  |
| [#25749](https://github.com/vllm-project/vllm/issues/25749) | [RFC]: Support Prefill Context Parallel (PCP) | pisceskkk | 2025-09-26 | 🟢 开放 | 一般 |  |  |
| [#26703](https://github.com/vllm-project/vllm/issues/26703) | [RFC]: [gpt-oss] Supporting Any MCP Server Tools | alecsolder | 2025-10-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-26703-rfc-gpt-oss-supporting-any-mcp-server-to.md) |
| [#26946](https://github.com/vllm-project/vllm/issues/26946) | [RFC]: Enable libtorch-ABI-stable vLLM cuda wheels | janeyx99 | 2025-10-15 | 🟢 开放 | 一般 |  |  |
| [#27691](https://github.com/vllm-project/vllm/issues/27691) | [RFC]: Tracking Spec Decode Support | MatthewBonanni | 2025-10-28 | ⚪ 已关闭 | 一般 |  |  |
| [#28326](https://github.com/vllm-project/vllm/issues/28326) | [RFC]: vLLM Support for Generic Model Definitions | bwasti | 2025-11-07 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-28326-rfc-vllm-support-for-generic-model-defin.md) |
| [#29295](https://github.com/vllm-project/vllm/issues/29295) | [RFC]: Support Dynamic Context Parallelism | Kurumi5210 | 2025-11-24 | ⚪ 已关闭 | 一般 |  |  |
| [#29485](https://github.com/vllm-project/vllm/issues/29485) | [RFC]: Detailed Usage Guide for AFD Support in vLLM | Alan-D-Chen | 2025-11-26 | ⚪ 已关闭 | 一般 |  |  |
| [#30038](https://github.com/vllm-project/vllm/issues/30038) | [RFC]: Add Peak Split & Tail Optimization Strategy | Pr0Wh1teGivee | 2025-12-04 | ⚪ 已关闭 | 一般 |  |  |
| [#32103](https://github.com/vllm-project/vllm/issues/32103) | [RFC]: Extending vLLM Pause/Resume API to support in-flight requests and DPEP scenarios | kouroshHakha | 2026-01-11 | ⚪ 已关闭 | 一般 |  |  |
| [#32219](https://github.com/vllm-project/vllm/issues/32219) | [RFC]: Add Helion integration in vLLM | gmagogsfm | 2026-01-12 | ⚪ 已关闭 | 一般 |  |  |
| [#32402](https://github.com/vllm-project/vllm/issues/32402) | [RFC]: Add a new `/collect_env` api to response current vllm instance environment | lengrongfu | 2026-01-15 | ⚪ 已关闭 | 一般 |  |  |
| [#32648](https://github.com/vllm-project/vllm/issues/32648) | [RFC]: Add --disable-inference flag deployments without GPU | hyeongyun0916 | 2026-01-20 | ⚪ 已关闭 | 一般 |  |  |
| [#33096](https://github.com/vllm-project/vllm/issues/33096) | [RFC]: Add protection against using deprecated or incorrect environment variables | gshtras | 2026-01-26 | ⚪ 已关闭 | 一般 |  |  |
| [#33599](https://github.com/vllm-project/vllm/issues/33599) | [RFC]: Improving vLLM Dependency Compatibility with Downstream Ecosystems | jeffreywang88 | 2026-02-02 | ⚪ 已关闭 | 一般 |  |  |
| [#35962](https://github.com/vllm-project/vllm/issues/35962) | [RFC]: Add PPL and KLD to VLLM | phaelon74 | 2026-03-04 | 🟢 开放 | 一般 |  |  |
| [#36091](https://github.com/vllm-project/vllm/issues/36091) | [RFC]: Add InstantTensor Support in vLLM | arlo-scitix | 2026-03-05 | ⚪ 已关闭 | 一般 |  |  |
| [#36155](https://github.com/vllm-project/vllm/issues/36155) | [RFC]: dLLM support via plugin (spec-decode path reuse) | AlonKellner-RedHat | 2026-03-05 | ⚪ 已关闭 | 一般 |  |  |
| [#36668](https://github.com/vllm-project/vllm/issues/36668) | [RFC]: Support Registry Mechanism for KVCacheSpec | MengqingCao | 2026-03-10 | ⚪ 已关闭 | 一般 |  |  |
| [#38231](https://github.com/vllm-project/vllm/issues/38231) | [RFC]: Support Dynamic Model Switching and Flexible Collective Communication in External … | KaisennHu | 2026-03-26 | ⚪ 已关闭 | 一般 |  |  |
| [#38651](https://github.com/vllm-project/vllm/issues/38651) | [RFC]: Add `max_tokens_per_doc` support for rerank and scoring endpoints | jefp | 2026-03-31 | ⚪ 已关闭 | 一般 |  |  |
| [#39433](https://github.com/vllm-project/vllm/issues/39433) | RFC: Add logit_scale to PoolerConfig for Affine Score Calibration (Platt Scaling) | jefp | 2026-04-09 | ⚪ 已关闭 | 一般 |  |  |
| [#39434](https://github.com/vllm-project/vllm/issues/39434) | [RFC]: Add logit_scale to PoolerConfig for Affine Score Calibration (Platt Scaling) | jefp | 2026-04-09 | ⚪ 已关闭 | 一般 |  |  |
| [#39504](https://github.com/vllm-project/vllm/issues/39504) | [RFC]: Enable prompt_embeds content parts in Chat Completions API | LuisRobaina | 2026-04-10 | ⚪ 已关闭 | 一般 |  |  |
| [#40244](https://github.com/vllm-project/vllm/issues/40244) | [RFC]: Add API to restore free_block_queue allocation order for long-running RLHF / rollo… | mrn3088 | 2026-04-18 | 🟢 开放 | 一般 |  |  |
| [#41843](https://github.com/vllm-project/vllm/issues/41843) | [RFC]: Add DeepStream as a video loader backend for GPU-accelerated Video decode | ViranjanPagar | 2026-05-06 | ⚪ 已关闭 | 一般 |  |  |
| [#45771](https://github.com/vllm-project/vllm/issues/45771) | [RFC]: Enable connector LoRA for audio multimodal models — convert nn.Linear projectors t… | anshulkulhari7 | 2026-06-16 | ⚪ 已关闭 | 一般 |  |  |
| [#46526](https://github.com/vllm-project/vllm/issues/46526) | [RFC]: Add Helion linear backend for vLLM | xiaohongchen1991 | 2026-06-23 | 🟢 开放 | 一般 |  |  |
| [#48148](https://github.com/vllm-project/vllm/issues/48148) | [RFC]: Remove Group Activation Ordering (actorder=group) Support from vLLM and Compressio… | kylesayrs | 2026-07-09 | 🟢 开放 | 一般 |  |  |
| [#49752](https://github.com/vllm-project/vllm/issues/49752) | [RFC]: vLLM Agentic Coding Readiness Survey | penli9 | 2026-07-24 | 🟢 开放 | 一般 |  |  |
| [#52567](https://github.com/vllm-project/vllm/issues/52567) | [RFC]: First-class agentic inference support in vLLM | franciscojavierarceo | 2026-08-17 | ⚪ 已关闭 | 一般 |  |  |
| [#52689](https://github.com/vllm-project/vllm/issues/52689) | [RFC]: support rl feature : weight checker | shiyuan680 | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#52817](https://github.com/vllm-project/vllm/issues/52817) | [RFC]: Hybrid SSM models and last-block-replay with SpecDec and APC enabled | roikoren755 | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#53916](https://github.com/vllm-project/vllm/issues/53916) | [RFC]: Native Text Watermarking Support | TQCB | 2026-08-26 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-53916-rfc-native-text-watermarking-support.md) |
| [#53933](https://github.com/vllm-project/vllm/issues/53933) | [RFC]: Elastic EP support in Model Runner V2 | almogtavor | 2026-08-26 | 🟢 开放 | 一般 |  |  |
| [#53938](https://github.com/vllm-project/vllm/issues/53938) | [RFC] Config-driven AITER auto-enable instead of a manual opt-in? | wjabbour | 2026-08-26 | ⚪ 已关闭 | 一般 |  |  |
| [#54443](https://github.com/vllm-project/vllm/issues/54443) | [RFC]: Weight version attribution for RL rollouts and Background survey | ShuoleiWang | 2026-08-30 | 🟢 开放 | 一般 |  |  |
| [#54940](https://github.com/vllm-project/vllm/issues/54940) | [RFC]: Support VLLM Container Snapshot with Checkpoint/Restore in Userspace (CRIU) | lbr711 | 2026-09-02 | 🟢 开放 | 一般 |  |  |
| [#55207](https://github.com/vllm-project/vllm/issues/55207) | [RFC]: Add QoL improvements to profiler | czhu-cohere | 2026-09-03 | ⚪ 已关闭 | 一般 |  |  |
| [#55267](https://github.com/vllm-project/vllm/issues/55267) | [RFC]: add uno spec decode | HF-001 | 2026-09-04 | 🟢 开放 | 一般 |  |  |
| [#55421](https://github.com/vllm-project/vllm/issues/55421) | [RFC] RL Test Consolidation: File Ownership, Coverage Docstrings, and CI Integration | Ronald1995 | 2026-09-05 | 🟢 开放 | 一般 |  |  |
| [#55916](https://github.com/vllm-project/vllm/issues/55916) | [RFC]: Add an RDNA4 FlyDSL all-reduce backend | big-yellow-duck | 2026-09-08 | 🟢 开放 | 一般 |  |  |
| [#57961](https://github.com/vllm-project/vllm/issues/57961) | [RFC]: Add native AliceAI model support | timka-byzov | 2026-09-21 | 🟢 开放 | 一般 |  |  |
| [#60931](https://github.com/vllm-project/vllm/issues/60931) | WIP [RFC]: SupportsReload, a single post-load contract and reload verifier (follow-up to … | hao-aaron | 2026-10-09 | 🟢 开放 | 一般 |  |  |
| [#60965](https://github.com/vllm-project/vllm/issues/60965) | [RFC]: Support Internally Managed Elastic EP with the MP DP Backend | RyanMa29 | 2026-10-10 | 🟢 开放 | 一般 |  |  |
| [#63004](https://github.com/vllm-project/vllm/issues/63004) | [RFC]: add eage4 spec decode | HF-001 | 2026-10-22 | 🟢 开放 | 一般 |  |  |
| [#65223](https://github.com/vllm-project/vllm/issues/65223) | [RFC]: TokenGate integration into vLLM | luis-gasparschroeder | 2026-11-02 | 🟢 开放 | 一般 |  |  |

## CI / Infra / Process(CI 与基础流程) (25)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#6801](https://github.com/vllm-project/vllm/issues/6801) | [RFC]: Performance Roadmap | simon-mo | 2024-07-25 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-6801-rfc-performance-roadmap.md) |
| [#8782](https://github.com/vllm-project/vllm/issues/8782) | [RFC]: Add Goodput Metric to Benchmark Serving | Imss27 | 2024-09-24 | ⚪ 已关闭 | 一般 |  |  |
| [#10818](https://github.com/vllm-project/vllm/issues/10818) | [RFC]: Disaggregated prefilling and KV cache transfer roadmap | KuntaiDu | 2024-12-02 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-10818-rfc-disaggregated-prefilling-and-kv-cach.md) |
| [#20265](https://github.com/vllm-project/vllm/issues/20265) | [RFC]: Online inference benchmark tool for multi-turn conversations | pliops-daniels | 2025-06-30 | ⚪ 已关闭 | 一般 |  |  |
| [#23033](https://github.com/vllm-project/vllm/issues/23033) | [RFC]: Support mmmu benchmark | tanruixiang | 2025-08-16 | ⚪ 已关闭 | 一般 |  |  |
| [#27084](https://github.com/vllm-project/vllm/issues/27084) | [RFC]: Add vllm benchs subcommand to benchmark test  multiple rounds | lengrongfu | 2025-10-17 | ⚪ 已关闭 | 一般 |  |  |
| [#30383](https://github.com/vllm-project/vllm/issues/30383) | [RFC]: Multi-Process Benchmark Architecture for Scaling Beyond Single-Core Limits | GaoHuaZhang | 2025-12-10 | ⚪ 已关闭 | 一般 |  |  |
| [#32029](https://github.com/vllm-project/vllm/issues/32029) | [RFC]: Online Quantization Roadmap | vkuzo | 2026-01-09 | ⚪ 已关闭 | 一般 |  |  |
| [#33640](https://github.com/vllm-project/vllm/issues/33640) | [RFC]: Enable reproducible benchmarking in benchmark_serving_multi_turn with API-usage to… | hyunnnchoi | 2026-02-03 | ⚪ 已关闭 | 一般 |  |  |
| [#33689](https://github.com/vllm-project/vllm/issues/33689) | [RFC]: KV Offloading Roadmap | orozery | 2026-02-03 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33689-rfc-kv-offloading-roadmap.md) |
| [#36948](https://github.com/vllm-project/vllm/issues/36948) | [RFC] [Feature]: PD Disaggregation with MooncakeConnector Roadmap | dtcccc | 2026-03-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-36948-rfc-feature-pd-disaggregation-with-moonc.md) |
| [#37979](https://github.com/vllm-project/vllm/issues/37979) | [RFC]: Intel Quantization Support Roadmap (H1 2026) | yiliu30 | 2026-03-24 | ⚪ 已关闭 | 一般 |  |  |
| [#39287](https://github.com/vllm-project/vllm/issues/39287) | [RFC]: Handle GDN prefill kernel JIT compilation failures - seeking community input | Alex-ai-future | 2026-04-08 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-39287-rfc-handle-gdn-prefill-kernel-jit-compil.md) |
| [#40902](https://github.com/vllm-project/vllm/issues/40902) | [Roadmap] DeepSeek V4 | ivanium | 2026-04-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-40902-roadmap-deepseek-v4.md) |
| [#45036](https://github.com/vllm-project/vllm/issues/45036) | [RFC]: [Roadmap] Mooncake Store Connector Feature Enrichment Roadmap | LCAIZJ | 2026-06-09 | 🟢 开放 | 一般 |  |  |
| [#45221](https://github.com/vllm-project/vllm/issues/45221) | [RFC]: AuxOutput Connector — block-keyed outputs and storage roadmap | huangyibo | 2026-06-11 | 🟢 开放 | 一般 |  |  |
| [#45668](https://github.com/vllm-project/vllm/issues/45668) | [Roadmap] Minimax M3 | gau-nernst | 2026-06-15 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-45668-roadmap-minimax-m3.md) |
| [#51214](https://github.com/vllm-project/vllm/issues/51214) | [RFC]: ARK H2 Upstream Roadmap for vLLM INC Path XPU Inference | Zhenzhong1 | 2026-08-06 | 🟢 开放 | 一般 |  |  |
| [#52095](https://github.com/vllm-project/vllm/issues/52095) | RFC: Kimi K3 MoonEP integration roadmap | kaijunli-infr | 2026-08-13 | 🟢 开放 | 一般 |  |  |
| [#52167](https://github.com/vllm-project/vllm/issues/52167) | [RFC]: Extended online quantization roadmap | fxmarty-amd | 2026-08-13 | 🟢 开放 | 一般 |  |  |
| [#55259](https://github.com/vllm-project/vllm/issues/55259) | [RFC]: Optional native startup self-benchmarking for forward-pass performance characteriz… | tedzhouhk | 2026-09-03 | 🟢 开放 | 一般 | 4 seconds to 9 seconds and a single worker pause stalled TP… |  |
| [#56968](https://github.com/vllm-project/vllm/issues/56968) | [RFC]: Plain-Python watermarking performance regression benchmark | SyaOtiLan | 2026-09-15 | 🟢 开放 | 一般 |  |  |
| [#58906](https://github.com/vllm-project/vllm/issues/58906) | [RFC] RL Weight Update Roadmap | aoshen02 | 2026-09-27 | 🟢 开放 | 一般 |  |  |
| [#59750](https://github.com/vllm-project/vllm/issues/59750) | [RFC]: Handle empty responses in synthetic acceptance benchmarks | whx-sjtu | 2026-10-02 | 🟢 开放 | 一般 |  |  |
| [#61698](https://github.com/vllm-project/vllm/issues/61698) | [RFC]: Bring-Your-Own-GPU (BYOG) reproducible evals and benchmarks for PRs | LucasWilkinson | 2026-10-14 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-61698-rfc-bring-your-own-gpu-byog-reproducible.md) |

## Other(其他) (267)

| RFC | 标题 | 作者 | 创建 | 状态 | 档位 | 收益声明 | 摘要卡 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#2526](https://github.com/vllm-project/vllm/issues/2526) | [RFC] Speedup vLLM inference with Intel@ Extension for PyTorch* | liangan1 | 2024-01-21 | ⚪ 已关闭 | 一般 | 47x compared with batch_size=1, whereas P90 token latency g… |  |
| [#3587](https://github.com/vllm-project/vllm/issues/3587) | [RFC]: Interface and Abstraction for Distributed Inference Environment | youkaichao | 2024-03-23 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-3587-rfc-interface-and-abstraction-for-distri.md) |
| [#4407](https://github.com/vllm-project/vllm/issues/4407) | [RFC]: environment variable management in vllm | youkaichao | 2024-04-27 | ⚪ 已关闭 | 一般 |  |  |
| [#4663](https://github.com/vllm-project/vllm/issues/4663) | [RFC]: Inline Golden (Expected) Tests | youkaichao | 2024-05-07 | ⚪ 已关闭 | 一般 |  |  |
| [#4926](https://github.com/vllm-project/vllm/issues/4926) | [RFC]: Postmerge performance suite | simon-mo | 2024-05-20 | ⚪ 已关闭 | 一般 |  |  |
| [#5377](https://github.com/vllm-project/vllm/issues/5377) | [RFC]: OpenVINO vLLM backend | ilya-lavrenov | 2024-06-10 | ⚪ 已关闭 | 一般 |  |  |
| [#5423](https://github.com/vllm-project/vllm/issues/5423) | [RFC]: Improve guided decoding (logit_processor) APIs and performance. | rkooo567 | 2024-06-11 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-5423-rfc-improve-guided-decoding-logit-proces.md) |
| [#5520](https://github.com/vllm-project/vllm/issues/5520) | [RFC]: Usage Data Enhancement for v0.5.* | simon-mo | 2024-06-14 | ⚪ 已关闭 | 一般 |  |  |
| [#5565](https://github.com/vllm-project/vllm/issues/5565) | [RFC]: Branch based version control, and development version | simon-mo | 2024-06-15 | ⚪ 已关闭 | 一般 |  |  |
| [#5716](https://github.com/vllm-project/vllm/issues/5716) | [RFC]: proper resource cleanup for LLM class with file-like usage | youkaichao | 2024-06-20 | ⚪ 已关闭 | 一般 |  |  |
| [#5816](https://github.com/vllm-project/vllm/issues/5816) | [RFC] Changes to CI workflow for PRs | khluu | 2024-06-25 | ⚪ 已关闭 | 一般 |  |  |
| [#5825](https://github.com/vllm-project/vllm/issues/5825) | [RFC]: Classifier-Free Guidance | Vermeille | 2024-06-25 | ⚪ 已关闭 | 一般 |  |  |
| [#6275](https://github.com/vllm-project/vllm/issues/6275) | [RFC]: Enhancing LoRA Management for Production Environments in vLLM | Jeffwan | 2024-07-10 | ⚪ 已关闭 | 一般 |  |  |
| [#6518](https://github.com/vllm-project/vllm/issues/6518) | (not planned) | afeldman-nm | 2024-07-17 | ⚪ 已关闭 | 一般 |  |  |
| [#6913](https://github.com/vllm-project/vllm/issues/6913) | [RFC]: Asynchronous Output Processor | megha95 | 2024-07-29 | ⚪ 已关闭 | 一般 |  |  |
| [#7131](https://github.com/vllm-project/vllm/issues/7131) | [RFC]: vLLM plugin system | youkaichao | 2024-08-04 | ⚪ 已关闭 | 一般 |  |  |
| [#7366](https://github.com/vllm-project/vllm/issues/7366) | [RFC]: Encoder/decoder models & feature compatibility | afeldman-nm | 2024-08-09 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-7366-rfc-encoder-decoder-models-feature-compa.md) |
| [#8279](https://github.com/vllm-project/vllm/issues/8279) | [RFC]: More functionality for API control | paulcx | 2024-09-09 | ⚪ 已关闭 | 一般 |  |  |
| [#8306](https://github.com/vllm-project/vllm/issues/8306) | [RFC]: Reimplement and separate beam search on top of vLLM core | youkaichao | 2024-09-09 | ⚪ 已关闭 | 一般 |  |  |
| [#9314](https://github.com/vllm-project/vllm/issues/9314) | [RFC]: Let every model be a reward model/embedding model for PRMs | zhuzilin | 2024-10-12 | ⚪ 已关闭 | 一般 |  |  |
| [#9669](https://github.com/vllm-project/vllm/issues/9669) | [RFC]: Model Deprecation Policy | youkaichao | 2024-10-24 | ⚪ 已关闭 | 一般 |  |  |
| [#10114](https://github.com/vllm-project/vllm/issues/10114) | [RFC]: Merge input processor and input mapper for multi-modal models | DarkLight1337 | 2024-11-07 | ⚪ 已关闭 | 一般 |  |  |
| [#10666](https://github.com/vllm-project/vllm/issues/10666) | [RFC]: Create `VllmState` to save immutable args in `VllmConfig` | MengqingCao | 2024-11-26 | ⚪ 已关闭 | 一般 |  |  |
| [#10674](https://github.com/vllm-project/vllm/issues/10674) | [RFC]: Make any vLLM model a pooling model | DarkLight1337 | 2024-11-26 | ⚪ 已关闭 | 一般 |  |  |
| [#11400](https://github.com/vllm-project/vllm/issues/11400) | [RFC]: Fully SPMD Execution for Offline Inference | eric-haibin-lin | 2024-12-21 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-11400-rfc-fully-spmd-execution-for-offline-inf.md) |
| [#12080](https://github.com/vllm-project/vllm/issues/12080) | [RFC]: BatchLLM for better shared prefix utilizing in offline scenarios | JensenFire | 2025-01-15 | ⚪ 已关闭 | 一般 |  |  |
| [#12174](https://github.com/vllm-project/vllm/issues/12174) | [RFC]: Distribute LoRA adapters across deployment | joerunde | 2025-01-17 | ⚪ 已关闭 | 一般 |  |  |
| [#12249](https://github.com/vllm-project/vllm/issues/12249) | [RFC]: Hidden states processor | DarkLight1337 | 2025-01-21 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-12249-rfc-hidden-states-processor.md) |
| [#12878](https://github.com/vllm-project/vllm/issues/12878) | [RFC]: Is IntermediateTensors shared across all request_ids or different for each request… | ishitamed19 | 2025-02-07 | ⚪ 已关闭 | 一般 |  |  |
| [#12977](https://github.com/vllm-project/vllm/issues/12977) | [RFC]: copy pynvml code into vllm codebase | youkaichao | 2025-02-09 | ⚪ 已关闭 | 一般 |  |  |
| [#13371](https://github.com/vllm-project/vllm/issues/13371) | [RFC]: Error executing method 'load_model'. This might cause deadlock in distributed exec… | BlackPottery1928 | 2025-02-17 | ⚪ 已关闭 | 一般 |  |  |
| [#13414](https://github.com/vllm-project/vllm/issues/13414) | [RFC]: Prompt logprobs + APC compatibility | afeldman-nm | 2025-02-17 | ⚪ 已关闭 | 一般 |  |  |
| [#14168](https://github.com/vllm-project/vllm/issues/14168) | [RFC]: Deprecate `max_num_generation_tokens` | vz-gh | 2025-03-04 | ⚪ 已关闭 | 一般 |  |  |
| [#14438](https://github.com/vllm-project/vllm/issues/14438) | [RFC]: Configurable multi-modal data for profiling | DarkLight1337 | 2025-03-07 | ⚪ 已关闭 | 一般 |  |  |
| [#14610](https://github.com/vllm-project/vllm/issues/14610) | [RFC]: A proper way to deal with 'Ray does not allocate any GPUs on the driver node' && '… | huiyeruzhou | 2025-03-11 | ⚪ 已关闭 | 一般 |  |  |
| [#14703](https://github.com/vllm-project/vllm/issues/14703) | [WIP][RFC]: Use auto-functionalization V2 in PyTorch 2.7+ | ProExpertProg | 2025-03-12 | 🟢 开放 | 一般 |  |  |
| [#14764](https://github.com/vllm-project/vllm/issues/14764) | [RFC]: Schema for checking input shapes for multi-modal models | DarkLight1337 | 2025-03-13 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-14764-rfc-schema-for-checking-input-shapes-for.md) |
| [#15810](https://github.com/vllm-project/vllm/issues/15810) | [RFC]: this object obtained from obj cache is not really used | senxiangms | 2025-03-31 | ⚪ 已关闭 | 一般 |  |  |
| [#15970](https://github.com/vllm-project/vllm/issues/15970) | [RFC]: AWS Neuron 2.23 NxD Inference with vLLM V0 | mrinalks | 2025-04-02 | ⚪ 已关闭 | 一般 |  |  |
| [#16012](https://github.com/vllm-project/vllm/issues/16012) | [RFC]: Is `huggingface-cli[hf_xet]` needed for vllm build? | Shafi-Hussain | 2025-04-03 | ⚪ 已关闭 | 一般 |  |  |
| [#16527](https://github.com/vllm-project/vllm/issues/16527) | [RFC]: How about change name specialized_manager to specialized_kv_cache_manager ? | princepride | 2025-04-12 | ⚪ 已关闭 | 一般 |  |  |
| [#16678](https://github.com/vllm-project/vllm/issues/16678) | [RFC]: tool_calls and None types. | elliott-marut | 2025-04-15 | ⚪ 已关闭 | 一般 |  |  |
| [#17067](https://github.com/vllm-project/vllm/issues/17067) | [RFC]: All Ops should be determined during init and wrapped in a Layer Module to avoid en… | tjtanaa | 2025-04-23 | 🟢 开放 | 一般 |  |  |
| [#17657](https://github.com/vllm-project/vllm/issues/17657) | Migrating from `yapf` to `ruff format` | hmellor | 2025-05-05 | ⚪ 已关闭 | 一般 |  |  |
| [#17720](https://github.com/vllm-project/vllm/issues/17720) | [RFC]: Enabling Arm Neoverse CI Runners | nikhil-arm | 2025-05-06 | ⚪ 已关闭 | 一般 |  |  |
| [#18571](https://github.com/vllm-project/vllm/issues/18571) | [RFC]: Deprecating vLLM V0 | WoosukKwon | 2025-05-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18571-rfc-deprecating-vllm-v0.md) |
| [#18826](https://github.com/vllm-project/vllm/issues/18826) | [RFC]: Controlling the maximum length of the waiting queue | chaunceyjiang | 2025-05-28 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-18826-rfc-controlling-the-maximum-length-of-th.md) |
| [#19376](https://github.com/vllm-project/vllm/issues/19376) | [RFC]: A Strategic Framework for Extensibility and Innovation in vLLM | yarongmu-google | 2025-06-09 | ⚪ 已关闭 | 一般 |  |  |
| [#19702](https://github.com/vllm-project/vllm/issues/19702) | [RFC]: Multimodal data IPC improvement | dongluw | 2025-06-16 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-19702-rfc-multimodal-data-ipc-improvement.md) |
| [#19886](https://github.com/vllm-project/vllm/issues/19886) | [RFC]: In-place weights loading | 22quinn | 2025-06-20 | ⚪ 已关闭 | 一般 |  |  |
| [#20476](https://github.com/vllm-project/vllm/issues/20476) | [RFC]: A developer friendly tool for multi-instance deployment with Ray | zhouyeju | 2025-07-04 | ⚪ 已关闭 | 一般 |  |  |
| [#20492](https://github.com/vllm-project/vllm/issues/20492) | [RFC]: KV-Cache Interoperability API Standardization | vMaroon | 2025-07-04 | ⚪ 已关闭 | 一般 |  |  |
| [#20687](https://github.com/vllm-project/vllm/issues/20687) | [RFC]: better differentiate infos, warnings and errors from exceptions | scwgoire | 2025-07-09 | ⚪ 已关闭 | 一般 |  |  |
| [#20708](https://github.com/vllm-project/vllm/issues/20708) | [RFC]: Continue on Device agnostic API abstraction to current_platform.XXX | xuechendi | 2025-07-09 | ⚪ 已关闭 | 一般 |  |  |
| [#20805](https://github.com/vllm-project/vllm/issues/20805) | [RFC]: EPLB Execution Optimization From pr 18343 | david6666666 | 2025-07-11 | ⚪ 已关闭 | 一般 |  |  |
| [#20947](https://github.com/vllm-project/vllm/issues/20947) | [RFC]: vLLM build locally should benefit from ccache | zou3519 | 2025-07-14 | ⚪ 已关闭 | 一般 |  |  |
| [#20962](https://github.com/vllm-project/vllm/issues/20962) | [RFC][FEATURE]: TTFT Routing | chickeyton | 2025-07-15 | ⚪ 已关闭 | 一般 |  |  |
| [#21047](https://github.com/vllm-project/vllm/issues/21047) | [RFC]: Migrate vLLM Docs back to Sphinx for Localization | hwhsu1231 | 2025-07-16 | ⚪ 已关闭 | 一般 |  |  |
| [#21113](https://github.com/vllm-project/vllm/issues/21113) | [RFC]: Reuse multimodal embeddings from encoder cache | knlnguyen1901 | 2025-07-17 | ⚪ 已关闭 | 一般 |  |  |
| [#21469](https://github.com/vllm-project/vllm/issues/21469) | [RFC]: Shorten all of the CI by reducing `cudagraph_capture_sizes` for most of the unit t… | tjtanaa | 2025-07-23 | ⚪ 已关闭 | 一般 |  |  |
| [#21475](https://github.com/vllm-project/vllm/issues/21475) | [RFC]: vLLM vs HuggingFace numerical parity report | 22quinn | 2025-07-23 | ⚪ 已关闭 | 一般 |  |  |
| [#21796](https://github.com/vllm-project/vllm/issues/21796) | [RFC]: Optimize embedding task | maxdebayser | 2025-07-29 | 🟢 开放 | 🔥 重点 | com/en/engineering-blog/embedding-inference-arctic-16x-fast… | [卡片](rfc-cards/rfc-21796-rfc-optimize-embedding-task.md) |
| [#21885](https://github.com/vllm-project/vllm/issues/21885) | - | hzh0425 | 2025-07-30 | ⚪ 已关闭 | 一般 |  |  |
| [#21995](https://github.com/vllm-project/vllm/issues/21995) | [RFC] Run HF processing on GPU | DarkLight1337 | 2025-07-31 | 🟢 开放 | 一般 |  |  |
| [#22041](https://github.com/vllm-project/vllm/issues/22041) | [RFC]: Reduce Unit Test to Speed Up CI | yewentao256 | 2025-07-31 | ⚪ 已关闭 | 一般 |  |  |
| [#22044](https://github.com/vllm-project/vllm/issues/22044) | [RFC]: Optimize Input Media Processing in vLLM | huachenheli | 2025-08-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22044-rfc-optimize-input-media-processing-in-v.md) |
| [#22082](https://github.com/vllm-project/vllm/issues/22082) | [RFC]: Clear and stable interface for platform in vLLM | wangxiyuan | 2025-08-01 | ⚪ 已关闭 | 一般 |  |  |
| [#22493](https://github.com/vllm-project/vllm/issues/22493) | [RFC]: Should the gpt-oss reasoning parser use harmony directly? | sethkimmel3 | 2025-08-08 | ⚪ 已关闭 | 一般 |  |  |
| [#22548](https://github.com/vllm-project/vllm/issues/22548) | [RFC]: | 123wujiao | 2025-08-08 | ⚪ 已关闭 | 一般 |  |  |
| [#22647](https://github.com/vllm-project/vllm/issues/22647) | [RFC]: Improve `vllm serve --help` Output for Enhanced User Experience | JustinXHale | 2025-08-11 | ⚪ 已关闭 | 一般 |  |  |
| [#22880](https://github.com/vllm-project/vllm/issues/22880) | [RFC]: Unified Input Formatting and Processing via `Renderer` | ywang96 | 2025-08-14 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22880-rfc-unified-input-formatting-and-process.md) |
| [#22932](https://github.com/vllm-project/vllm/issues/22932) | [RFC]: Path to vLLM 1.0 Release | robertgshaw2-redhat | 2025-08-14 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-22932-rfc-path-to-vllm-1-0-release.md) |
| [#23113](https://github.com/vllm-project/vllm/issues/23113) | [RFC]: Enabling Multiple Graphs Based on pre-defined conditions | ilmarkov | 2025-08-18 | ⚪ 已关闭 | 一般 |  |  |
| [#23354](https://github.com/vllm-project/vllm/issues/23354) | [RFC] How could we prevent model like R1 E2E accuracy down to 0? | yewentao256 | 2025-08-21 | ⚪ 已关闭 | 一般 |  |  |
| [#23431](https://github.com/vllm-project/vllm/issues/23431) | [RFC]: Continuous profiling and regression prevention for vLLM | linzebing | 2025-08-22 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-23431-rfc-continuous-profiling-and-regression.md) |
| [#23446](https://github.com/vllm-project/vllm/issues/23446) | [RFC]: Redesigning Persistent Batch in vLLM | WoosukKwon | 2025-08-22 | ⚪ 已关闭 | 一般 |  |  |
| [#23474](https://github.com/vllm-project/vllm/issues/23474) | [RFC]: Disallow extra vocab for LoRA | WoosukKwon | 2025-08-24 | ⚪ 已关闭 | 一般 |  |  |
| [#23892](https://github.com/vllm-project/vllm/issues/23892) | [RFC]: Remove LoRA bias | jeejeelee | 2025-08-29 | ⚪ 已关闭 | 一般 |  |  |
| [#24384](https://github.com/vllm-project/vllm/issues/24384) | [RFC]: Decoupling vLLM Configuration from Hugging Face | charlotte12l | 2025-09-07 | ⚪ 已关闭 | 一般 |  |  |
| [#24603](https://github.com/vllm-project/vllm/issues/24603) | [RFC]: Responses API full functionality without stores | alecsolder | 2025-09-10 | ⚪ 已关闭 | 一般 |  |  |
| [#24685](https://github.com/vllm-project/vllm/issues/24685) | [RFC]: Decode Context Parallel for GQA | frank-wei | 2025-09-11 | ⚪ 已关闭 | 一般 |  |  |
| [#24885](https://github.com/vllm-project/vllm/issues/24885) | [RFC] Clarifying vLLM Shutdown Semantics | markmc | 2025-09-15 | ⚪ 已关闭 | 一般 |  |  |
| [#25144](https://github.com/vllm-project/vllm/issues/25144) | [RFC]: vLLM’s Path to Sustainable CI/CD | yeqcharlotte | 2025-09-18 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-25144-rfc-vllm-s-path-to-sustainable-ci-cd.md) |
| [#25212](https://github.com/vllm-project/vllm/issues/25212) | [RFC]: Renaming vllm metric name | can-anyscale | 2025-09-18 | ⚪ 已关闭 | 一般 |  |  |
| [#25670](https://github.com/vllm-project/vllm/issues/25670) | [RFC]: Migration from buildkite to GHA | jeanschmidt | 2025-09-25 | ⚪ 已关闭 | 一般 |  |  |
| [#25700](https://github.com/vllm-project/vllm/issues/25700) | [RFC]: Limit the use of envvars in vLLM | zhuohan123 | 2025-09-25 | 🟢 开放 | 一般 |  |  |
| [#26118](https://github.com/vllm-project/vllm/issues/26118) | [RFC]: Migrate from Ubuntu 20.04 as Build Base to manylinux | simon-mo | 2025-10-02 | ⚪ 已关闭 | 一般 |  |  |
| [#26293](https://github.com/vllm-project/vllm/issues/26293) | [RFC]: replace deprecated torch_dtype by dtype | yannicks1 | 2025-10-06 | ⚪ 已关闭 | 一般 |  |  |
| [#26934](https://github.com/vllm-project/vllm/issues/26934) | [RFC][ResponsesAPI]: Separating State & Providing Flexibility for serving ResponsesAPI | qandrew | 2025-10-15 | ⚪ 已关闭 | 一般 |  |  |
| [#27653](https://github.com/vllm-project/vllm/issues/27653) | [RFC]: include past-reasoning for harmony(gpt-oss) formatting in chat completions API | dawid-sliwa | 2025-10-28 | ⚪ 已关闭 | 一般 |  |  |
| [#27661](https://github.com/vllm-project/vllm/issues/27661) | [RFC]: Consolidated tool call parser implementations by type (JSON, Python, XML, Harmony) | bbrowning | 2025-10-28 | ⚪ 已关闭 | 一般 |  |  |
| [#27755](https://github.com/vllm-project/vllm/issues/27755) | [RFC]: `reasoning_content` -> `reasoning` | hmellor | 2025-10-29 | ⚪ 已关闭 | 一般 |  |  |
| [#27767](https://github.com/vllm-project/vllm/issues/27767) | [RFC]: Coordinating vLLM and PyTorch Release Timelines. Starting with PyTorch Release 2.10 | atalman | 2025-10-29 | ⚪ 已关闭 | 一般 |  |  |
| [#27866](https://github.com/vllm-project/vllm/issues/27866) | [RFC]: Internal Process-level Fault Tolerance for vLLM | fangyuchu | 2025-10-31 | 🟢 开放 | 一般 |  |  |
| [#27995](https://github.com/vllm-project/vllm/issues/27995) | [RFC]: Make PassConfig flags less verbose | ProExpertProg | 2025-11-03 | ⚪ 已关闭 | 一般 |  |  |
| [#28071](https://github.com/vllm-project/vllm/issues/28071) | [RFC]: Pin all dependencies | njhill | 2025-11-04 | ⚪ 已关闭 | 一般 |  |  |
| [#28163](https://github.com/vllm-project/vllm/issues/28163) | [RFC]: SageMaker Day-0 Compatibility with vLLM | zhaozuy | 2025-11-05 | ⚪ 已关闭 | 一般 |  |  |
| [#28190](https://github.com/vllm-project/vllm/issues/28190) | [RFC]: Optimizing the Expand and Shrink Algorithms for LoRA Inference in Dense Models | caozuoba | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#28226](https://github.com/vllm-project/vllm/issues/28226) | [RFC]: Redesign Logprobs data structure to reduce GC cost | Jialin | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#28239](https://github.com/vllm-project/vllm/issues/28239) | [RFC]: Replace list[list[int]] to numpy arrays to mitigate GC overhead. | Jialin | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#28243](https://github.com/vllm-project/vllm/issues/28243) | [RFC]: Externally managed elastic EP (decouple from Ray backend) | galletas1712 | 2025-11-06 | ⚪ 已关闭 | 一般 |  |  |
| [#28261](https://github.com/vllm-project/vllm/issues/28261) | [RFC]: Responses API Tools Type Simplification | alecsolder | 2025-11-07 | ⚪ 已关闭 | 一般 |  |  |
| [#28465](https://github.com/vllm-project/vllm/issues/28465) | [RFC]: Publishing KVBlocks event with mm_hash | Bounty-hunter | 2025-11-11 | ⚪ 已关闭 | 一般 |  |  |
| [#29362](https://github.com/vllm-project/vllm/issues/29362) | [RFC]: Resettle examples. | noooop | 2025-11-25 | ⚪ 已关闭 | 一般 |  |  |
| [#29640](https://github.com/vllm-project/vllm/issues/29640) | [RFC]: Confusing name in FusedMoE | ijpq | 2025-11-28 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-29640-rfc-confusing-name-in-fusedmoe.md) |
| [#30839](https://github.com/vllm-project/vllm/issues/30839) | [RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC | brandonpelfrey | 2025-12-17 | 🟢 开放 | 一般 |  |  |
| [#30930](https://github.com/vllm-project/vllm/issues/30930) | [Draft] [RFC]: vLLM + UCC Backend | lengrongfu | 2025-12-18 | ⚪ 已关闭 | 一般 |  |  |
| [#31229](https://github.com/vllm-project/vllm/issues/31229) | [RFC]: Early-Fail Tokenization Guard for Completions or Chat Completions | scratch-ml | 2025-12-23 | ⚪ 已关闭 | 一般 |  |  |
| [#31249](https://github.com/vllm-project/vllm/issues/31249) | [RFC]: Improve environment variable declaration and handling | ProExpertProg | 2025-12-24 | 🟢 开放 | 一般 |  |  |
| [#31803](https://github.com/vllm-project/vllm/issues/31803) | [RFC]: Video Frame Sparsification via Pixel-Level Similarity for Efficient Multimodal Lon… | rayleeku | 2026-01-06 | ⚪ 已关闭 | 一般 |  |  |
| [#31848](https://github.com/vllm-project/vllm/issues/31848) | [RFC]: Native Weight Syncing APIs | hao-aaron | 2026-01-07 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-31848-rfc-native-weight-syncing-apis.md) |
| [#32200](https://github.com/vllm-project/vllm/issues/32200) | [RFC]: Change `kv_load_failure_policy` Default from "recompute" to "fail" | NickLucche | 2026-01-12 | ⚪ 已关闭 | 一般 |  |  |
| [#32218](https://github.com/vllm-project/vllm/issues/32218) | [RFC]: Consolidate Multimodal Related Info | charlotte12l | 2026-01-12 | ⚪ 已关闭 | 一般 |  |  |
| [#32358](https://github.com/vllm-project/vllm/issues/32358) | [RFC]: vLLM IR: A Functional Intermediate Representation for vLLM | ProExpertProg | 2026-01-14 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32358-rfc-vllm-ir-a-functional-intermediate-re.md) |
| [#32613](https://github.com/vllm-project/vllm/issues/32613) | [RFC]: More robust model accuracy testing with configurable and tiered coverage | xinli-sw | 2026-01-19 | ⚪ 已关闭 | 一般 |  |  |
| [#32713](https://github.com/vllm-project/vllm/issues/32713) | [RFC]: Unified Parser for reasoning, tool calling | qandrew | 2026-01-20 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-32713-rfc-unified-parser-for-reasoning-tool-ca.md) |
| [#32850](https://github.com/vllm-project/vllm/issues/32850) | [RFC]: Clarify policy for Open Responses API extensions in vLLM | DanielMe | 2026-01-22 | ⚪ 已关闭 | 一般 |  |  |
| [#33118](https://github.com/vllm-project/vllm/issues/33118) | [RFC]: Hidden States Extraction | fynnsu | 2026-01-26 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-33118-rfc-hidden-states-extraction.md) |
| [#33381](https://github.com/vllm-project/vllm/issues/33381) | [RFC]: Align with the openresponses spec. | chaunceyjiang | 2026-01-30 | ⚪ 已关闭 | 一般 |  |  |
| [#33445](https://github.com/vllm-project/vllm/issues/33445) | [RFC] Remove mandatory ray installation | yewentao256 | 2026-01-30 | ⚪ 已关闭 | 一般 |  |  |
| [#33478](https://github.com/vllm-project/vllm/issues/33478) | [RFC]: Introduce ATOM as model implementation backend for AMD GPU | zejunchen-zejun | 2026-01-31 | ⚪ 已关闭 | 一般 |  |  |
| [#33789](https://github.com/vllm-project/vllm/issues/33789) | [RFC]: How will vLLM adapt to more diverse KVCacheSpecs? | sheep94lion | 2026-02-04 | ⚪ 已关闭 | 一般 |  |  |
| [#33805](https://github.com/vllm-project/vllm/issues/33805) | [RFC]: Expose RequestOutput hook for programmatic use of Serving layer | alecsolder | 2026-02-04 | ⚪ 已关闭 | 一般 |  |  |
| [#34001](https://github.com/vllm-project/vllm/issues/34001) | [RFC] Change the directory layout from `scaled_mm/` and `mixed_precision/` to backend-fir… | tjtanaa | 2026-02-06 | ⚪ 已关闭 | 一般 |  |  |
| [#34018](https://github.com/vllm-project/vllm/issues/34018) | [RFC]: Helix (Context + Tensor) Parallelism for Efficient Long-Context Decoding | sungsooha | 2026-02-06 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-34018-rfc-helix-context-tensor-parallelism-for.md) |
| [#34303](https://github.com/vllm-project/vllm/issues/34303) | [RFC]: CUDA Checkpoint/Restore for Near-Zero Cold Starts | elizabetht | 2026-02-11 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-34303-rfc-cuda-checkpoint-restore-for-near-zer.md) |
| [#34333](https://github.com/vllm-project/vllm/issues/34333) | [RFC]: Automated baselining and degradation detection | jmkuebler | 2026-02-11 | 🟢 开放 | 一般 |  |  |
| [#35516](https://github.com/vllm-project/vllm/issues/35516) | [RFC]: why block_hash maps not always a single KVCacheBlock. | binbinzhm | 2026-02-27 | ⚪ 已关闭 | 一般 |  |  |
| [#35639](https://github.com/vllm-project/vllm/issues/35639) | [RFC]: `vllm bench eval` for Unified Accuracy + Performance Evaluation | AndreasKaratzas | 2026-03-01 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-35639-rfc-vllm-bench-eval-for-unified-accuracy.md) |
| [#35848](https://github.com/vllm-project/vllm/issues/35848) | [RFC]: Revamp Ray Distributed Executor Backend (from Ray team) | jeffreywang88 | 2026-03-03 | ⚪ 已关闭 | 🔥 重点 |  | [卡片](rfc-cards/rfc-35848-rfc-revamp-ray-distributed-executor-back.md) |
| [#35908](https://github.com/vllm-project/vllm/issues/35908) | [RFC]: Model-specific realtime streaming abstraction | TheCodeWrangler | 2026-03-03 | ⚪ 已关闭 | 一般 |  |  |
| [#36406](https://github.com/vllm-project/vllm/issues/36406) | [RFC]: Changing the Default Behavior of usage Field in Responses | Csrayz | 2026-03-08 | ⚪ 已关闭 | 一般 |  |  |
| [#36998](https://github.com/vllm-project/vllm/issues/36998) | [RFC]: Observation Plugin for Intercepting & Routing on Activations | DDDDarrenWB | 2026-03-13 | 🟢 开放 | 一般 |  |  |
| [#37003](https://github.com/vllm-project/vllm/issues/37003) | [RFC]: Context-Aware KV-Cache Retention API (Prioritized Evictions) | vMaroon | 2026-03-13 | 🟢 开放 | 一般 |  |  |
| [#37075](https://github.com/vllm-project/vllm/issues/37075) | [RFC]: Opt-in Media URL Cache for `MediaConnector` | AndreasKaratzas | 2026-03-14 | ⚪ 已关闭 | 一般 |  |  |
| [#37849](https://github.com/vllm-project/vllm/issues/37849) | [RFC]: Unify the function of getting device count | wincent8 | 2026-03-23 | ⚪ 已关闭 | 一般 |  |  |
| [#38141](https://github.com/vllm-project/vllm/issues/38141) | [RFC] Streaming Video Input for Real-Time Video Understanding | lishunyang12 | 2026-03-25 | ⚪ 已关闭 | 一般 |  |  |
| [#38147](https://github.com/vllm-project/vllm/issues/38147) | [RFC]: Endpoints to Retrieve User Configuration and Runtime Data | hickeyma | 2026-03-25 | ⚪ 已关闭 | 一般 |  |  |
| [#39076](https://github.com/vllm-project/vllm/issues/39076) | [RFC]: Entropy-Gated Online KV Block Expiration During Active Decode | groot-code24 | 2026-04-06 | ⚪ 已关闭 | 一般 |  |  |
| [#39158](https://github.com/vllm-project/vllm/issues/39158) | [RFC][Test]: Unified Platform-Aware Test Skip Mechanism | jikunshang | 2026-04-07 | ⚪ 已关闭 | 一般 |  |  |
| [#39370](https://github.com/vllm-project/vllm/issues/39370) | [RFC][vLLM IR] `rms_norm` weight passing inconsistency | ProExpertProg | 2026-04-09 | ⚪ 已关闭 | 一般 |  |  |
| [#39694](https://github.com/vllm-project/vllm/issues/39694) | [RFC]:  PR de-dup/Similarity-Check  CI workflow ? | panpan0000 | 2026-04-13 | 🟢 开放 | 一般 |  |  |
| [#39701](https://github.com/vllm-project/vllm/issues/39701) | [RFC] Replace routing replay with CUDA-graph-compatible device cache approach | TomerBN-Nvidia | 2026-04-13 | ⚪ 已关闭 | 一般 |  |  |
| [#39829](https://github.com/vllm-project/vllm/issues/39829) | [RFC]: Reduce CI Cost by Pruning Low-Value Unit Tests | yewentao256 | 2026-04-14 | ⚪ 已关闭 | 一般 |  |  |
| [#39848](https://github.com/vllm-project/vllm/issues/39848) | [RFC]: Unifying Tool Calling via Region-Scoped Guided Decoding, Tool-Aware Grammars, and … | bitbottrap | 2026-04-15 | 🟢 开放 | 一般 |  |  |
| [#39871](https://github.com/vllm-project/vllm/issues/39871) | [RFC]: Replace Hardcoded Device Strings with current_platform and Implement Linting | wincent8 | 2026-04-15 | ⚪ 已关闭 | 一般 |  |  |
| [#39884](https://github.com/vllm-project/vllm/issues/39884) | [RFC]: Automatic test target determination for CI | khluu | 2026-04-15 | ⚪ 已关闭 | 一般 |  |  |
| [#39942](https://github.com/vllm-project/vllm/issues/39942) | [RFC]: Fault-Tolerant EP Collectives and Fault-Aware EPLB | jeffreywang88 | 2026-04-15 | ⚪ 已关闭 | 一般 |  |  |
| [#39979](https://github.com/vllm-project/vllm/issues/39979) | [RFC]: Ultimate Better Observability. | noooop | 2026-04-16 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-39979-rfc-ultimate-better-observability.md) |
| [#40533](https://github.com/vllm-project/vllm/issues/40533) | [RFC]: Hybrid checkpoint ABI for non-KV prefix resume | CarlOskarRost | 2026-04-21 | ⚪ 已关闭 | 一般 |  |  |
| [#40567](https://github.com/vllm-project/vllm/issues/40567) | [RFC]: Fault tolerant EPLB | tzulingk | 2026-04-22 | ⚪ 已关闭 | 一般 |  |  |
| [#40620](https://github.com/vllm-project/vllm/issues/40620) | [RFC]: Unified Device Capability Abstraction for Cross-Platform Feature Detection | jikunshang | 2026-04-22 | ⚪ 已关闭 | 一般 |  |  |
| [#40628](https://github.com/vllm-project/vllm/issues/40628) | [RFC][vLLM IR]: Batch Invariance Dispatching in vLLM IR | ProExpertProg | 2026-04-22 | ⚪ 已关闭 | 一般 |  |  |
| [#40846](https://github.com/vllm-project/vllm/issues/40846) | [RFC]: Rust front-end | njhill | 2026-04-24 | 🟢 开放 | 🔥 重点 | Default Python saturates at only 19% of Rust throughput wit… | [卡片](rfc-cards/rfc-40846-rfc-rust-front-end.md) |
| [#41821](https://github.com/vllm-project/vllm/issues/41821) | [RFC]: Adaptive throughput/latency profile for RL rollout long-tail | aoshen02 | 2026-05-06 | ⚪ 已关闭 | 一般 |  |  |
| [#42082](https://github.com/vllm-project/vllm/issues/42082) | [RFC]: Standardize KV-cache Layouts | LucasWilkinson | 2026-05-08 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42082-rfc-standardize-kv-cache-layouts.md) |
| [#42164](https://github.com/vllm-project/vllm/issues/42164) | [RFC]:  Introducing State Management in vLLM IR System | Alex-ai-future | 2026-05-09 | ⚪ 已关闭 | 一般 |  |  |
| [#42361](https://github.com/vllm-project/vllm/issues/42361) | [RFC]: Zero-copy LoRA loading from tmpfs via mmap + cudaHostRegister | ultism | 2026-05-12 | ⚪ 已关闭 | 一般 |  |  |
| [#42484](https://github.com/vllm-project/vllm/issues/42484) | [RFC] Production-boundary measurement on H100 + vLLM 0.19.1: throughput plateau at c=4→16… | jacob-sunho-kim | 2026-05-13 | ⚪ 已关闭 | 一般 |  |  |
| [#42515](https://github.com/vllm-project/vllm/issues/42515) | [RFC]: External Elastic EP Scaling | lqh-hh | 2026-05-13 | 🟢 开放 | 一般 |  |  |
| [#42659](https://github.com/vllm-project/vllm/issues/42659) | [RFC]: Per-Layer Parallelism Policy | sungsooha | 2026-05-14 | 🟢 开放 | 一般 |  |  |
| [#42770](https://github.com/vllm-project/vllm/issues/42770) | [RFC]: Changes in vLLM Model Development | WoosukKwon | 2026-05-15 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-42770-rfc-changes-in-vllm-model-development.md) |
| [#43702](https://github.com/vllm-project/vllm/issues/43702) | [RFC]: Non-blocking core model loop | njhill | 2026-05-26 | ⚪ 已关闭 | 一般 |  |  |
| [#43711](https://github.com/vllm-project/vllm/issues/43711) | [RFC]: Debug observation for prefix-cache lookup boundary before request-level stats | gustavgauge | 2026-05-26 | ⚪ 已关闭 | 一般 |  |  |
| [#43918](https://github.com/vllm-project/vllm/issues/43918) | [RFC]:  Move trainer-side weight transfer logic out of `vllm` | SumanthRH | 2026-05-28 | ⚪ 已关闭 | 一般 |  |  |
| [#43999](https://github.com/vllm-project/vllm/issues/43999) | [RFC] External post-generation classifier hook API | geodesia-ai | 2026-05-29 | 🟢 开放 | 一般 |  |  |
| [#44211](https://github.com/vllm-project/vllm/issues/44211) | [RFC]: vLLM NaN Reporting | elvircrn | 2026-06-01 | ⚪ 已关闭 | 一般 |  |  |
| [#44276](https://github.com/vllm-project/vllm/issues/44276) | [RFC]: Model customized KVCache Planning | MengqingCao | 2026-06-02 | ⚪ 已关闭 | 一般 |  |  |
| [#44996](https://github.com/vllm-project/vllm/issues/44996) | [RFC]: Fast Runtime Resize of a Pre-Built EP Topology | nanookfyf | 2026-06-09 | ⚪ 已关闭 | 一般 |  |  |
| [#45098](https://github.com/vllm-project/vllm/issues/45098) | [RFC]: Image Token Pruning for Multimodal Models | garrygale | 2026-06-10 | ⚪ 已关闭 | 一般 |  |  |
| [#45585](https://github.com/vllm-project/vllm/issues/45585) | [RFC] RL CI Matrix for vLLM: Behavioral + Physical + Protocol Coverage | aoshen02 | 2026-06-14 | ⚪ 已关闭 | 🔥 重点 | - [x] CUDA graph pool VMM slowdown regression (3-5x with gr… | [卡片](rfc-cards/rfc-45585-rfc-rl-ci-matrix-for-vllm-behavioral-phy.md) |
| [#45702](https://github.com/vllm-project/vllm/issues/45702) | [RFC]: Partial Cache Hits for Hybrid Models | ZJY0516 | 2026-06-15 | ⚪ 已关闭 | 一般 |  |  |
| [#45815](https://github.com/vllm-project/vllm/issues/45815) | [RFC]: Optimizing vLLM for High Beam-Width Generative Recommendation Inference | ehaleva | 2026-06-16 | ⚪ 已关闭 | 一般 |  |  |
| [#46185](https://github.com/vllm-project/vllm/issues/46185) | RFC: State-Aware Monolingual Reasoning Constraints for DeepSeek-R1 Models (Opt-In) | Himanshu-is-code | 2026-06-19 | 🟢 开放 | 一般 |  |  |
| [#46358](https://github.com/vllm-project/vllm/issues/46358) | [RFC]: Unify Context Parallelism in vLLM (PCP ⊥ DCP) | JaredforReal | 2026-06-22 | ⚪ 已关闭 | 一般 |  |  |
| [#46565](https://github.com/vllm-project/vllm/issues/46565) | [RFC]: Entrypoint plugins. | noooop | 2026-06-24 | ⚪ 已关闭 | 一般 |  |  |
| [#46722](https://github.com/vllm-project/vllm/issues/46722) | [RFC]: Reduce multimodal payload in token-in/token-out by deferring `pixel_values` prepro… | hyeongyun0916 | 2026-06-25 | 🟢 开放 | 一般 |  |  |
| [#46937](https://github.com/vllm-project/vllm/issues/46937) | [RFC]: TorchComms DeviceCommunicator for VLLM | jemitche1 | 2026-06-28 | 🟢 开放 | 一般 |  |  |
| [#47264](https://github.com/vllm-project/vllm/issues/47264) | [RFC]: Qwen3-ASR pseudo-streaming can repeat or miss segments | shiverking | 2026-07-01 | 🟢 开放 | 一般 |  |  |
| [#47456](https://github.com/vllm-project/vllm/issues/47456) | [RFC]: Design a shared warmup infrastructure for JITs | LopezCastroRoberto | 2026-07-02 | 🟢 开放 | 一般 |  |  |
| [#47582](https://github.com/vllm-project/vllm/issues/47582) | [RFC]: Opt-in incremental prompt encoding for multi-turn chat | bird | 2026-07-03 | 🟢 开放 | 一般 |  |  |
| [#47812](https://github.com/vllm-project/vllm/issues/47812) | [RFC]: Precise group-aware recovery for KV load failures on hybrid models | ClockOfDestiny | 2026-07-07 | 🟢 开放 | 一般 |  |  |
| [#48038](https://github.com/vllm-project/vllm/issues/48038) | [RFC]: PyTorch vLLM correctness audit | morrison-turnansky | 2026-07-08 | 🟢 开放 | 一般 |  |  |
| [#48049](https://github.com/vllm-project/vllm/issues/48049) | [RFC][Feature]: First-class session id for conversation aware routing | karen-sy | 2026-07-08 | 🟢 开放 | 一般 |  |  |
| [#48201](https://github.com/vllm-project/vllm/issues/48201) | [RFC]: GLM-5.x thinking_token_budget defaults and regression tests | AlanJager | 2026-07-10 | 🟢 开放 | 一般 |  |  |
| [#48227](https://github.com/vllm-project/vllm/issues/48227) | [RFC]: Standardize vLLM Entrypoint Error Handling | zqzten | 2026-07-10 | 🟢 开放 | 一般 |  |  |
| [#48305](https://github.com/vllm-project/vllm/issues/48305) | [RFC] Training-Inference Consistency for RL | aoshen02 | 2026-07-11 | 🟢 开放 | 一般 |  |  |
| [#48306](https://github.com/vllm-project/vllm/issues/48306) | [RFC] RL Information Retrieval: APIs & Introspection | aoshen02 | 2026-07-11 | ⚪ 已关闭 | 一般 |  |  |
| [#48310](https://github.com/vllm-project/vllm/issues/48310) | [RFC] Sleep/Wake Correctness for RL | aoshen02 | 2026-07-11 | 🟢 开放 | 一般 |  |  |
| [#48312](https://github.com/vllm-project/vllm/issues/48312) | [RFC] Weight Reload Correctness for RL | aoshen02 | 2026-07-11 | 🟢 开放 | 一般 |  |  |
| [#48478](https://github.com/vllm-project/vllm/issues/48478) | [RFC] Fail-Closed Graph Storage Contract for Weight Reload | aoshen02 | 2026-07-13 | ⚪ 已关闭 | 一般 |  |  |
| [#48501](https://github.com/vllm-project/vllm/issues/48501) | [RFC]: Session-centric KV-cache orchestration over typed session identity | vMaroon | 2026-07-13 | 🟢 开放 | 一般 |  |  |
| [#48540](https://github.com/vllm-project/vllm/issues/48540) | [RFC]: manylinux compatibility baseline for aarch64 binary dependencies | brandonpelfrey | 2026-07-13 | 🟢 开放 | 一般 |  |  |
| [#48743](https://github.com/vllm-project/vllm/issues/48743) | [RFC]: Return extracted hidden states in the generation response | Davidnet | 2026-07-15 | 🟢 开放 | 一般 |  |  |
| [#49047](https://github.com/vllm-project/vllm/issues/49047) | [RFC]: Standalone Rust renderer | sagearc | 2026-07-18 | ⚪ 已关闭 | 一般 |  |  |
| [#49702](https://github.com/vllm-project/vllm/issues/49702) | [RFC]: Expose narrow EPLB extension hooks for out-of-tree accelerators | freyfwt | 2026-07-24 | 🟢 开放 | 一般 |  |  |
| [#50079](https://github.com/vllm-project/vllm/issues/50079) | RL Weight Update Tracker | aoshen02 | 2026-07-28 | ⚪ 已关闭 | 一般 |  |  |
| [#50127](https://github.com/vllm-project/vllm/issues/50127) | [RFC]: Shared Context Parallelism: peer-addressable GPU objects for PCP and DCP | foraxe | 2026-07-28 | 🟢 开放 | 一般 |  |  |
| [#50834](https://github.com/vllm-project/vllm/issues/50834) | [RFC]:  Unify the Tensor Type for Device-Pointer Storage to `torch.uint64` | chaojun-zhang | 2026-08-03 | 🟢 开放 | 一般 |  |  |
| [#51275](https://github.com/vllm-project/vllm/issues/51275) | [RFC]: Race-free port management: pick ports at bind time, publish over an existing chann… | aoshen02 | 2026-08-06 | 🟢 开放 | 一般 |  |  |
| [#51476](https://github.com/vllm-project/vllm/issues/51476) | [RFC]: DP pause/sleep correctness: coordinator latch, request admission, device-idle comp… | aoshen02 | 2026-08-08 | 🟢 开放 | 一般 |  |  |
| [#51940](https://github.com/vllm-project/vllm/issues/51940) | [RFC]: Per-request activation of multiple LoRA adapters with static scales | KisuYang | 2026-08-12 | 🟢 开放 | 一般 |  |  |
| [#51963](https://github.com/vllm-project/vllm/issues/51963) | [RFC]: Report characters/second alongside tokens/second in vllm bench serve | spped2000 | 2026-08-12 | 🟢 开放 | 一般 |  |  |
| [#52777](https://github.com/vllm-project/vllm/issues/52777) | [RFC]: Rank-Skew-Aware Collective Attribution for Tensor-Parallel Serving | calvin-fei | 2026-08-18 | 🟢 开放 | 一般 |  |  |
| [#52916](https://github.com/vllm-project/vllm/issues/52916) | [RFC] Adaptive spin grace + bounded arch waits (aarch64 WFET, x86 WAITPKG) for shm_broadc… | randomvariable | 2026-08-19 | 🟢 开放 | 一般 |  |  |
| [#52959](https://github.com/vllm-project/vllm/issues/52959) | [RFC]: Internal State Checkpoints for Mamba Align Mode | ZJY0516 | 2026-08-19 | 🟢 开放 | 一般 |  |  |
| [#53178](https://github.com/vllm-project/vllm/issues/53178) | [RFC]: Experimental Recirculation for causal decoder models | mihir-s-05 | 2026-08-21 | ⚪ 已关闭 | 一般 |  |  |
| [#53194](https://github.com/vllm-project/vllm/issues/53194) | [RFC]: A conformance suite for KV-cache key partitioning | Tejasvichand | 2026-08-21 | 🟢 开放 | 一般 |  |  |
| [#53343](https://github.com/vllm-project/vllm/issues/53343) | [RFC]: Sleep Mode Tensor Ownership and Recovery | Ronald1995 | 2026-08-22 | 🟢 开放 | 一般 |  |  |
| [#53371](https://github.com/vllm-project/vllm/issues/53371) | [RFC]: Optimization: Take over a partial-hit block instead of CoW copy when ref_cnt == 1 | THUqliu | 2026-08-22 | 🟢 开放 | 一般 |  |  |
| [#53401](https://github.com/vllm-project/vllm/issues/53401) | [RFC]: Experimental Recirculation for causal decoders | mihir-s-05 | 2026-08-22 | 🟢 开放 | 一般 |  |  |
| [#53421](https://github.com/vllm-project/vllm/issues/53421) | [RFC]: First-Class, Orchestrator-Agnostic KV Hint Envelope for Agentic Workloads | karen-sy | 2026-08-23 | 🟢 开放 | 一般 |  |  |
| [#53485](https://github.com/vllm-project/vllm/issues/53485) | [RFC]: Reconcile Backpressure Admission | Srinivasoo7 | 2026-08-23 | 🟢 开放 | 一般 |  |  |
| [#53563](https://github.com/vllm-project/vllm/issues/53563) | [RFC]: SM90 Exact Fused DSA Prefill Indexer | fxfxfxfxfxfxfxfx | 2026-08-24 | 🟢 开放 | 🔥 重点 | 10x indexer-stage geometric-mean speedup over the target va… | [卡片](rfc-cards/rfc-53563-rfc-sm90-exact-fused-dsa-prefill-indexer.md) |
| [#53684](https://github.com/vllm-project/vllm/issues/53684) | [RFC]: Allow the DSA Indexer to Use Context Parallelism Independently | fxfxfxfxfxfxfxfx | 2026-08-25 | 🟢 开放 | 一般 | Our preliminary operator benchmarks on 2x NVIDIA H20 report… |  |
| [#53788](https://github.com/vllm-project/vllm/issues/53788) | [RFC]: Using Helion for Selected vLLM CustomOps | yushangdi | 2026-08-25 | 🟢 开放 | 一般 |  |  |
| [#54477](https://github.com/vllm-project/vllm/issues/54477) | [RFC]: Direct Weight Reload for RL Training | aoshen02 | 2026-08-31 | 🟢 开放 | 一般 |  |  |
| [#54644](https://github.com/vllm-project/vllm/issues/54644) | [RFC]: Preemption victim-selection extension point (least-computed-first as the first bui… | shivasathishs-rp | 2026-08-31 | ⚪ 已关闭 | 一般 |  |  |
| [#54653](https://github.com/vllm-project/vllm/issues/54653) | [RFC] Organize the Python package by architectural domain | xiaoyanzi191 | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#54749](https://github.com/vllm-project/vllm/issues/54749) | [RFC]: The dynamic-SD K decision is a batch-keyed array index — agreeing on its shape bef… | seongyun1104 | 2026-09-01 | 🟢 开放 | 一般 |  |  |
| [#54864](https://github.com/vllm-project/vllm/issues/54864) | [RFC]: Truncate mode for `thinking_token_budget` | walterbm-cohere | 2026-09-02 | 🟢 开放 | 一般 |  |  |
| [#55487](https://github.com/vllm-project/vllm/issues/55487) | [RFC] Publish typed extra_keys in BlockStored KV events | zhouyou9505 | 2026-09-05 | ⚪ 已关闭 | 一般 |  |  |
| [#55639](https://github.com/vllm-project/vllm/issues/55639) | [RFC][EPD] Avoid redundant encoder work when Prefill can reuse KV or embeddings | jiaran-king | 2026-09-07 | 🟢 开放 | 一般 |  |  |
| [#55986](https://github.com/vllm-project/vllm/issues/55986) | [RFC] Multi-channel target audio ignores channel_reduction in normalize_audio | AdaAibaby | 2026-09-09 | 🟢 开放 | 一般 |  |  |
| [#56123](https://github.com/vllm-project/vllm/issues/56123) | [RFC]: P/D transfer contract across different parallel layouts with cp interleave size | GirasoleY | 2026-09-09 | 🟢 开放 | 一般 |  |  |
| [#56128](https://github.com/vllm-project/vllm/issues/56128) | RFC: PCIe-native all-reduce for TP on peer-capable, NVLink-less multi-GPU nodes | Zeuss5 | 2026-09-09 | ⚪ 已关闭 | 一般 |  |  |
| [#56133](https://github.com/vllm-project/vllm/issues/56133) | [RFC]: Runtime prefill/decode role switching for EP groups | Etelis | 2026-09-09 | 🟢 开放 | 一般 |  |  |
| [#56727](https://github.com/vllm-project/vllm/issues/56727) | [RFC] CLI configuration without runtime initialization | matteso1 | 2026-09-13 | 🟢 开放 | 一般 |  |  |
| [#56768](https://github.com/vllm-project/vllm/issues/56768) | [RFC] Exact single-session HOT continuation for hybrid Mamba models | KonnyakuMatcha | 2026-09-14 | ⚪ 已关闭 | 一般 |  |  |
| [#56860](https://github.com/vllm-project/vllm/issues/56860) | [RFC]: Per-position target token logprobs for prefill scoring | aoshen02 | 2026-09-14 | 🟢 开放 | 一般 |  |  |
| [#56916](https://github.com/vllm-project/vllm/issues/56916) | [RFC]: Windowed Hidden-State Collection for vLLM Rollouts | vx120 | 2026-09-15 | 🟢 开放 | 一般 |  |  |
| [#57111](https://github.com/vllm-project/vllm/issues/57111) | [RFC]: Checkpoint-aware cache eviction and segmented recomputation for hybrid models | warriorsniu | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#57200](https://github.com/vllm-project/vllm/issues/57200) | [RFC] Model console logging configuration as CLI/serve configuration | markmc | 2026-09-16 | 🟢 开放 | 一般 |  |  |
| [#57383](https://github.com/vllm-project/vllm/issues/57383) | [RFC]: Asymmetric P/D Deployment for DeepSeek-V4.1 Flash | gz944367214 | 2026-09-17 | 🟢 开放 | 一般 |  |  |
| [#57649](https://github.com/vllm-project/vllm/issues/57649) | [RFC]: Declarative cross-platform capability negotiation and effective configuration repo… | hlin99 | 2026-09-19 | 🟢 开放 | 一般 |  |  |
| [#57738](https://github.com/vllm-project/vllm/issues/57738) | [RFC]: Encoder-Only Prefill in P/D deployment for DeepSeek V4.1 Flash | hczboy | 2026-09-20 | 🟢 开放 | 一般 |  |  |
| [#58253](https://github.com/vllm-project/vllm/issues/58253) | [RFC]: Increase vLLM Review Capacity with a Reviewer Role | LucasWilkinson | 2026-09-23 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-58253-rfc-increase-vllm-review-capacity-with-a.md) |
| [#58537](https://github.com/vllm-project/vllm/issues/58537) | [RFC]: KV Context Editing — Relaxed Prefix Alignment for KV Allocation and Reuse | yxkyong | 2026-09-24 | 🟢 开放 | 一般 |  |  |
| [#58697](https://github.com/vllm-project/vllm/issues/58697) | [RFC]: Shared JSON Schema structure for structured-output checks and transformations | equiluxe | 2026-09-25 | 🟢 开放 | 一般 |  |  |
| [#58774](https://github.com/vllm-project/vllm/issues/58774) | [RFC]: Native span pooling for contextual chunk embeddings | LinzeShi | 2026-09-25 | 🟢 开放 | 一般 |  |  |
| [#58909](https://github.com/vllm-project/vllm/issues/58909) | [Feature]: [RFC] Dynamic Latent-Trajectory Consensus Governor for Reasoning Models (DeepS… | Arnab758 | 2026-09-27 | 🟢 开放 | 一般 |  |  |
| [#59038](https://github.com/vllm-project/vllm/issues/59038) | [RFC]: DeepSeek-V4.1-Flash performance on MI325X (gfx942) | frida-andersson | 2026-09-28 | 🟢 开放 | 一般 |  |  |
| [#59382](https://github.com/vllm-project/vllm/issues/59382) | [RFC]: MoRIIO WRITE failure handling and safe KV block reclamation | yuzhouo7 | 2026-09-30 | 🟢 开放 | 一般 |  |  |
| [#59542](https://github.com/vllm-project/vllm/issues/59542) | [RFC]: Opt-in return of the last hidden state at sampled positions, per request (Model Ru… | aminry | 2026-10-01 | 🟢 开放 | 一般 |  |  |
| [#59665](https://github.com/vllm-project/vllm/issues/59665) | [RFC]: Fast-Track Merging for Model Optimization PRs | benchislett | 2026-10-01 | 🟢 开放 | 🔥 重点 |  | [卡片](rfc-cards/rfc-59665-rfc-fast-track-merging-for-model-optimiz.md) |
| [#59708](https://github.com/vllm-project/vllm/issues/59708) | [RFC]: Modular and Extensible APIs & Execution Surface for Orchestrator-Driven KV Hint Po… | karen-sy | 2026-10-01 | 🟢 开放 | 一般 |  |  |
| [#60366](https://github.com/vllm-project/vllm/issues/60366) | [RFC]: Bounded prefix reuse for target-token prefill scoring | LinzeShi | 2026-10-07 | 🟢 开放 | 一般 |  |  |
| [#60422](https://github.com/vllm-project/vllm/issues/60422) | [RFC]:Ownership boundaries for externally orchestrated checkpoint and restore | leeZardNav | 2026-10-07 | 🟢 开放 | 一般 |  |  |
| [#60559](https://github.com/vllm-project/vllm/issues/60559) | [RFC]: TPSP Fusion Design | hlin99 | 2026-10-08 | 🟢 开放 | 一般 |  |  |
| [#60722](https://github.com/vllm-project/vllm/issues/60722) | [RFC]: Strcutured Outputs Not Respected When Reasoning Implicitly Ends | yzong-rh | 2026-10-08 | ⚪ 已关闭 | 一般 |  |  |
| [#62145](https://github.com/vllm-project/vllm/issues/62145) | [RFC]: Request-scoped auxiliary decision heads on a shared generative backbone | sylvanus4 | 2026-10-17 | 🟢 开放 | 一般 |  |  |
| [#62689](https://github.com/vllm-project/vllm/issues/62689) | [RFC]: Layered prefill: split prefill across decoder layer groups | gunjunlee | 2026-10-20 | 🟢 开放 | 一般 |  |  |
| [#62810](https://github.com/vllm-project/vllm/issues/62810) | [RFC] Expose allocator-issued KV block generations to asynchronous connectors | feichai0017 | 2026-10-21 | 🟢 开放 | 一般 |  |  |
| [#63336](https://github.com/vllm-project/vllm/issues/63336) | [RFC]: MoRIIO transfer contract: safe KV reclamation and bounded failure for P/D (READ an… | tjtanaa | 2026-10-23 | 🟢 开放 | 一般 |  |  |
| [#63359](https://github.com/vllm-project/vllm/issues/63359) | [RFC]: Use session Agent Hints to reuse prompt tokens and block hashes | RichealYoung | 2026-10-23 | 🟢 开放 | 一般 |  |  |
| [#63501](https://github.com/vllm-project/vllm/issues/63501) | [RFC]: Select the MRV2 DBO microbatch count per step within an allocated capacity | algebra-MCX | 2026-10-24 | 🟢 开放 | 一般 |  |  |
| [#63624](https://github.com/vllm-project/vllm/issues/63624) | [RFC]: Bring vLLM's development cost back to linear: a small core with owned, independent… | aijanai | 2026-10-25 | 🟢 开放 | 一般 |  |  |
| [#63734](https://github.com/vllm-project/vllm/issues/63734) | [RFC]: Define a graceful-draining contract for external Kubernetes controllers | raravind007 | 2026-10-26 | 🟢 开放 | 一般 |  |  |
| [#64099](https://github.com/vllm-project/vllm/issues/64099) | [RFC]: Consistent semantics for drafted, verified and accepted spec decode tokens | tomasruizt | 2026-10-27 | 🟢 开放 | 一般 |  |  |
| [#64380](https://github.com/vllm-project/vllm/issues/64380) | [RFC]: R3 delivery protocol: keys, sharing scope, write/read/release contract | aoshen02 | 2026-10-28 | 🟢 开放 | 一般 |  |  |
| [#64508](https://github.com/vllm-project/vllm/issues/64508) | [RFC]: Checkpoint-internal paging for hybrid recurrent state | kang-dongwook | 2026-10-29 | 🟢 开放 | 一般 |  |  |
| [#65056](https://github.com/vllm-project/vllm/issues/65056) | [RFC]: Selected input-row hidden-state export for full-vocabulary OPD teachers | NolenLiang | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#65121](https://github.com/vllm-project/vllm/issues/65121) | [RFC]: DBO with the allgather_reducescatter all2all backend on CUDA | algebra-MCX | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#65172](https://github.com/vllm-project/vllm/issues/65172) | [RFC]: Holemiss: Selective Recovery of Partially Lost LLM Cache State | valvarl | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#65255](https://github.com/vllm-project/vllm/issues/65255) | [RFC]: Out-of-tree KV-export connectors that hand a vLLM prefill to MLX (EXO) decode on A… | padwowen | 2026-11-02 | 🟢 开放 | 一般 |  |  |
| [#65502](https://github.com/vllm-project/vllm/issues/65502) | [RFC]: First-class, trusted forwarded prompt token IDs on the chat, Messages, and Respons… | ahg-g | 2026-11-03 | 🟢 开放 | 一般 |  |  |
| [#65913](https://github.com/vllm-project/vllm/issues/65913) | [RFC]: Deprecate and remove the torchao reload decorator in favor of reload_weights | aoshen02 | 2026-11-05 | 🟢 开放 | 一般 |  |  |

## 数据快照(下次更新的基线)

- 快照记录时间: 2026-11-07T13:53:26+00:00
- **vllm** @1ecc9cc4ce66 (upstream HEAD 2026-11-07) — commit: https://github.com/vllm-project/vllm/commit/1ecc9cc4ce6626352df39e4364579ec376c10761 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 917 → 重点档 222 → 摘要卡 222
- **sglang** @ba5906471c62 (upstream HEAD 2026-11-07) — commit: https://github.com/sgl-project/sglang/commit/ba5906471c62e82c0594a617b4d07d9b2c0fd2b3 | API 抓取窗口: 2026-11-07T12:33:17+00:00 ~ 2026-11-07T12:33:17+00:00 | RFC 202 → 重点档 42 → 摘要卡 42
