# RFC #56880: [RFC]: Add NCP-OLMo support and NCP DFlash speculative decoding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56880](https://github.com/vllm-project/vllm/issues/56880) |
| 作者 | LuckySJTU |
| 创建日期 | 2026-09-14 |
| 状态 | 🟢 open |
| labels | quantization, speculative-decoding |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
提议在 vLLM 内落地 NCP-OLMo 模型族后端并叠加 NCP DFlash 投机解码,核心难点是 request 级 HLM 层次状态契约。

## 提案动机
NCP-ArchPreview 在 chunk-level 高层模型(HLM)外围放 token 级 encoder/decoder 塔,公开 checkpoint 为 `NCPOlmo3ForCausalLM`。支持它不只是注册常规 decoder:HLM 持有 request-local 状态(不完整 encoder fragment、per-layer HLM states 与 K/V、selected concepts、routed decoder features),须跨 scheduler 行重排、完成、refill、chunked prefill 与 recompute preemption 绑定 request ID;prefix caching 只有连同 HLM snapshot 一起缓存才正确。Stage 2 v1 模型另有配套 NCP DFlash checkpoint,需要 target-feature capture、request 持有的 proposal context 与拒绝后事务性 rollback/replay。作者检索未找到现有实现。

## 提出的方案
两个 PR:
1. target backend:注册架构、加载 HF checkpoint(含 split Q/K/V、gate/up 融合映射),实现 token encoder、HLM、路由码本与 token decoder;token 塔用常规 vLLM Attention,HLM state 经 V2 model-state lifecycle 按 request ID 管理;首版 fail-closed(TP=1、PP=1、eager、无 prefix caching、无 CUDA graph)。
2. DFlash 扩展:复用 `method="dflash"`,新增 NCP speculator——按 request ID 缓存 proposal 侧 context K/V、按完成/refill 后 batch 推导 proposal 宽度、拒绝后事务性截断/恢复重放。默认 `sequential_exact`;`intra_chunk_exact` 限当前安全 HLM chunk;跨 chunk `segmented_kv_approx` 显式实验性。

## 优化点与收益
(单张 H200 合成快照、greedy、seed 42、3 次取中位)batch 1:8:target-only 27.05 tok/s,exact 29.19(1.079x,验收 89.10%),实验 adaptive 51.68(1.910x,49.46%);batch 8:2:120.86 / 117.31(0.971x)/ 135.29(1.119x,80.67%)。连续 refill(16 请求、max_num_seqs=8):target-only 68.11,exact 66.39(0.975x),adaptive 81.17(1.192x)。测试:target 55 通过、DFlash 88 通过;17 个非 DFlash checkpoint 均符合契约、同为 842 weight keys。

## 现状与落地
open,反馈期至少一周;实现与验证为 AI 辅助、提交者人工复核;两个分支已可运行,PR 将在 rebase 后提交;就 in-tree 还是扩展 out-of-tree hooks、两 PR 拆分边界、DFlash 配置归属、fail-closed prefix caching 征求维护者意见。

## 关联
HF 模型族 https://huggingface.co/collections/ArchSpace-Collection/ncp-archpreview;DFlash checkpoint https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_DFlash2_NCPFlash;target backend https://github.com/vllm-project/vllm/compare/main...LuckySJTU:vllm:dev/ncp-archpreview;DFlash 扩展 https://github.com/LuckySJTU/vllm/compare/dev/ncp-archpreview...dev/ncp-archpreview-dflash

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
