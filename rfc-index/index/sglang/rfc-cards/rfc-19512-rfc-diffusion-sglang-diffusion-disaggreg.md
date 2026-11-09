# RFC #19512: [RFC] [Diffusion] SGLang Diffusion Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19512](https://github.com/sgl-project/sglang/issues/19512) |
| 作者 | FredHuang99 |
| 创建日期 | 2026-02-27 |
| 状态 | ⚪ closed (2026-04-17) |
| labels | (none) |
| 热度 | 👍 4 · 💬 5 |
| 分类 | Disaggregated Serving |

## 一句话结论
把 LLM PD 分离经验推广到扩散模型:T2V 管线拆成 Encoder/Denoising/Decoder 三角色独立部署与调优。

## 提案动机
PD 分离在 LLM 上已证明高效;扩散模型尤其是 Text-to-Video 生成中,Text-Encoder、DiT、VAE 等管线模块间执行时间差异巨大,单体部署迫使异构阶段共享统一硬件配置,造成严重计算瓶颈。分离后可对各模块独立调并行策略与 GPU 配额,最大化 SM(Streaming Multiprocessor)利用率与整体管线吞吐。

## 提出的方案
在 SGLang 成熟的 LLM PD 分离架构与 Mooncake 数据传输基础上改造:
- 三角色拆分:EncoderRole(InputValidation/TextEncoding/Conditioning/TimestepPreparation/LatentPreparation 等阶段)、DenoisingRole、DecoderRole;Pipeline 按角色动态调整 _required_config_modules 与 create_pipeline_stages,只加载必要权重。
- TransferBuffer(CPU 暂存区)分两类:TransferMetaBuffer 存轻量非张量元数据(类比 PD 的 MetadataBuffer);TransferTensorBuffer 为重型张量(latents 等)的 pinned 内存池,由 Allocator 以 buddy 式 split/aggregate/coalesce 动态分配、释放时合并碎片;容量按角色独立配置,以吸收 Encoder 快、Denoising 慢的速度失配。
- TransferringQueue/SwappingQueue + TransferManager:维护 TransferEngine 并注册内存地址;receive/send event loop 增量缓存下游路由(Instance→Rank→IP:Port+Buffer 地址),线程池并发传输,transfer_slice 适配 FSDP/SP 等不同并行策略;Transfer.Poll 做请求状态同步(类似 KV.Poll)。
- DiffusionServer 全局调度器:FreeBufferSlots 记账 + 每角色 TryToAdd(TTA)队列,dispatch_event_loop 按 MaxFreeSlotsFirst/RoundRobin 等策略选实例,回调机制增减配额。
- 调度器 3-stream(H2D/Compute/D2H)CUDA 流重叠:torch.cuda.Event 同步,event.query() 非阻塞检查,网络传输按请求粒度触发(与 batch 边界解耦),避免 CPU 阻塞与管线气泡。
- 远期(Future Map):etcd 只做服务发现,数据面走 P2P 去中心化路由,FreeBufferSlots 经 ACK 捎带同步,消除中心调度器瓶颈与 SPOF。

## 优化点与收益
正文未给出量化数字(定性目标:最大化 SM 利用率与整体管线吞吐)。

## 现状与落地
已关闭(2026-04-17)。正文 TODO 清单(DiffusionServer、TransferManager、TransferBuffer、DisaggUtils 各项)均未勾选,看不出是否已实现,落地情况需查关联 PR。

## 关联
正文未列出(文中架构图为 GitHub 附件)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
