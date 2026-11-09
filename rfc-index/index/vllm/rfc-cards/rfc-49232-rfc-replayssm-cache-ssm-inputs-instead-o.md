# RFC #49232: [RFC]: ReplaySSM: cache SSM inputs instead of state for faster standard and speculative decode (Mamba2 + GDN)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49232](https://github.com/vllm-project/vllm/issues/49232) |
| 作者 | askliar |
| 创建日期 | 2026-07-20 |
| 状态 | 🟢 open |
| labels | RFC, stale |
| 热度 | 👍 3 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
提议 ReplaySSM:对 SSM/线性注意力解码只缓存近期 SSM 输入而非每步读写全部状态,靠 checkpoint + 环形缓冲重放,降低内存带宽开销并支持 O(1) 回滚投机解码。

## 提案动机
SSM decode 是内存带宽受限的:当前每一步都要读出并写回完整的循环状态。ReplaySSM 是针对 SSM 与线性注意力层 decode kernel 的内存带宽优化,由 Dao AI Lab 的 @Johnny-Liou 与 NVIDIA 的 @hnover-nv 独立并行开发(TensorRT-LLM #14203 为跨框架同期实现)。本 RFC 同时是 ReplaySSM 在 vLLM、FlashInfer、TensorRT-LLM 的追踪中心,取代 RFC #47572。两个受益场景:标准 decode——状态每 L 步写回一次而非每步,大致减半该受限 kernel 的主要内存流量;投机解码——SSM 状态更新不可逆,现实现要为每个草稿 token 保留独立状态(约 `(1 + num_spec)×` 状态),改为缓存输入后回滚变成环形缓冲指针移动(O(1) 而非 O(T)),并可一次性验证草稿窗口而非串行递推。收益在高 batch(算力充分内存受限)与混合模型短中等输出长度(SSM 层主导延迟)时最大。

## 提出的方案
- 保留单一 checkpoint 状态 + 环形缓冲缓存近期 SSM 输入;非 flush 步直接由 checkpoint 加缓存输入计算输出,每 `L` 步才写回一次完整状态。
- 完整实现先以单个大 draft PR(vllm#47576)落地供端到端评审,再拆分为小的自包含 PR 合入。
- 范围:Mamba2 与 Gated DeltaNet(GDN)层;标准与投机解码;初始 PR 即支持 stochastic rounding 与 prefix caching;新 Triton kernel + vLLM 集成(mixer dispatch、attention-metadata 游标、state-shape 尺寸),conv 路径复用现有 `causal_conv1d_update`;选定架构接入 FlashInfer CUDA kernel。
- Config flags(全部默认关闭):`--use-replayssm`(Mamba2/GDN 标准缓存解码)、`--use-replayssm-spec`(投机缓存解码,需投机解码且 `--mamba-cache-mode none`)、`--replayssm-buffer-len`(历史缓冲长度,默认 16)。关闭时 decode 路径与基线逐字节一致。
- 集成方式:ReplaySSM 缓存作为现有 paged mamba state 上的额外张量,复用 vLLM 的 paged-block 分配与 block-table 索引(continuous batching、逐请求分配/逐出、prefix caching);投机解码用两次 device 路由的 launch(verify 与 flush),flush 决策与回滚在 device 上计算,每个 batch size 一张捕获的 CUDA graph 即可覆盖。环形缓冲大小为 `B + T`,当 `history + T > B` 时 flush(B 为缓冲长度,T = num_spec_tokens + 1)。
- 双 kernel 轨道:Triton(vLLM,覆盖更广 SSM 类型)与 CUDA(FlashInfer,针对 Mamba2/GDN 调优,flashinfer#3975、#3324);kernel↔runtime 契约(接受 token 数、CUDA-graph 兼容、缓存形状/dtype)固定,可替换任意合规 kernel;方法与精度正交,可跨 FP16/BF16/FP8/INT8 移植。
- 路线图:短期为 Mamba2/GDN 标准解码(vllm#48018 进行中)与投机解码;中期为量化 SSM 状态格式、FlashInfer CUDA kernel、Mamba3/KDA/GDP 等;长期为量化 + SR(FP16、FP8 block-scale、INT8 block-scale)、Hopper/DGX Spark 支持等。

## 优化点与收益
测试条件:FP32 state、buffer length 16,整模型每步加速比(基线/ReplaySSM):Nemotron-Nano-4B(Mamba2, H100)bs 1/32/128/512 为 0.99/1.17/1.40/1.55;Qwen3.5-4B(GDN, H100)为 0.99/1.08/1.21/1.31;Nemotron-Super-120B(Mamba2, B300)为 0.98/1.10/1.27/1.67;Qwen3.5-122B(GDN, B300)为 0.98/1.07/1.14/1.20。投机解码:端到端吞吐,`(B, T) = (16, 4)`、B300、MTP drafter;`c/std` = ReplaySSM-spec / baseline-spec,`c/AR` = ReplaySSM-spec / no-spec autoregressive:Nemotron-Super-120B(Mamba2)c/std bs 1/32/256 为 1.09/1.07/1.88,c/AR 为 2.12/1.90/2.21;Qwen3.5-122B(GDN)c/std 为 0.99/1.19/1.80,c/AR 为 1.90/1.93/1.85。

## 现状与落地
仍在讨论中(open),作为跨 vLLM/FlashInfer/TensorRT-LLM 的追踪中心,状态表随工作落地持续更新:短期各项(Mamba2/GDN 标准解码、kernel↔runtime 契约、投机缓存解码)均 🔄 进行中,量化格式与更多 SSM 类型 ⏳ 计划中;反馈期自发布起一周,讨论活跃则延长。

## 关联
- 取代的 RFC: https://github.com/vllm-project/vllm/issues/47572
- 主 draft PR: https://github.com/vllm-project/vllm/pull/47576
- 首个 PR: https://github.com/vllm-project/vllm/pull/48018
- 博客: https://dao-lab.ai/blog/2026/replayssm/
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM/pull/14203 、FlashInfer: flashinfer#3975 / #3324

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
