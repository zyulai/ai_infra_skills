# RFC #53563: [RFC]: SM90 Exact Fused DSA Prefill Indexer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53563](https://github.com/vllm-project/vllm/issues/53563) |
| 作者 | fxfxfxfxfxfxfxfx |
| 创建日期 | 2026-08-24 |
| 状态 | 🟢 open |
| labels | RFC, quantization |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |

## 一句话结论
提议为 DeepSeek V3.2 的 SparseAttnIndexer prefill 增加可选 SM90 后端:单 kernel 融合 FP8 打分与精确 TopK,免去全量 logits 落 HBM。

## 提案动机
当前 prefill indexer 流程为 gather K -> fp8_fp4_mqa_logits -> 物化 logits -> top_k_per_row_prefill。长上下文下,全量 logits 张量的写入/读回与 kernel 边界开销显著:在作者当前 H20 矩阵中,`H=1M, B=32, Q=512` 场景峰值 score chunk 达 508.25 MiB,一次 indexer 调用累计物化 64.03 GiB 的 scores,而最终 TopK 索引输出仅 128 MiB。

## 提出的方案
- 新后端接入现有 `sparse_attn_indexer()` prefill 分支,只替换"打分 + TopK"阶段,输出仍进现有 `topk_indices_buffer`,下游 Sparse MLA attention 不变;decode、KV-cache 布局、K gather 均不动。
- 配置项:`attention_config.indexer_prefill_backend = "default" | "sm90_exact"`,默认保持现实现;不满足 GPU/dtype/H/D/topk/DCP/PCP/共享内存条件时报初始化错误。
- 首 PR 范围刻意收窄:SM90a(H20、H800、H100 分别验证)、DeepSeek V3.2 FP8 prefill、H64、D=128、topk=2048、单 GPU(无 DCP/PCP)、INT32 输出 `[M, 2048]` 无序;H32 将在独立 PR 中支持 GLM-5.1/5.2。DCP 不支持,因现 DCP merge 需要本地 TopK 分数而该 kernel 只输出索引。
- 内核设计(一查询一 CTA 流水线):TMA K/K-scale 双缓冲 → 4 个 WGMMA math warp groups(每轮 256 keys)→ 带代际标记的共享内存 score ring → FP32 ordered-bit histogram/radix cutoff → 运行中 Top-2048 与最终索引;768 线程/CTA、两个 TMA stage、每选择组 12,544 个 scores、232,448 字节动态共享内存(每 SM 一个 CTA on H20)、生产 cubin 零 stack/local memory;上游版将用静态 CUDA 构建,无首请求 JIT。
- 精确性契约:`exact` 指对融合 kernel 产生的 FP32 分数算出精确 cutoff(2048 个唯一索引,必含 score > c、不含 score < c、平票补齐);不声称与现有打分器 bitwise 等价(WGMMA/DeepGEMM 归约顺序不同)。

## 优化点与收益
初步算子级结果(单卡 NVIDIA H20,合成 FP8 输入,固定 H64、D=128、topk=2048;基线为 vLLM 0.27.1 的 FP32 `fp8_fp4_mqa_logits` + `top_k_per_row_prefill`,665 种形状:history KV 4K-1M × batch 1/4/16/32/128 × query 8-8192;两组结果来自独立运行,非配对回归):
- 665 种形状上对基线的几何平均加速 **1.3918981x**;
- 665 形状中 604 种快于 vLLM;
- 58/58 正确性用例通过,零采样 DeepGEMM cutoff 亏损。
正文另提议实施证据门槛:目标验证矩阵上 indexer 阶段几何平均加速 ≥ 1.10x,且两个长上下文配置 TTFT 改善 ≥ 3%。最新 main、H800/H100、真实模型、CUDA Graph 与端到端 TTFT 验证仍待完成。

## 现状与落地
状态 open,反馈期一周。处于征求维护者意见阶段:是否先接受 H64 SM90 融合 prefill indexer 作为 opt-in 实验后端、H32 另行 PR;PR 1(H64/DeepSeek V3.2)在提交评审前将补齐 cutoff oracle、自定义 op schema、CUDA Graph/compute-sanitizer 覆盖、与最新 main 的配对对比及真实 DeepSeek V3.2 prefill/MRCR/NIAH 测试。

## 关联
- [#52696](https://github.com/vllm-project/vllm/pull/52696)(FP16 logits + 原生 FP16 TopK)
- [#48726](https://github.com/vllm-project/vllm/pull/48726) 与 LiteTopK 论文(https://arxiv.org/abs/2607.11976,SM100 方向)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
