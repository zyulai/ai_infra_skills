# RFC #66163: [RFC][ROCm]: Batch-invariant kernels for Kimi-K3.1 on MI355X (KDA, DSA/indexer, MXFP4 MoE, LoRA, all-reduce)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66163](https://github.com/vllm-project/vllm/issues/66163) |
| 作者 | indianspeedster |
| 创建日期 | 2026-11-06 |
| 状态 | 🟢 open |
| labels | batch-invariance, kimi, rocm |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |

## 一句话结论
为 MI355X 上的 Kimi-K3.1 逐项落地 ROCm batch-invariant 内核(KDA/DSA/MXFP4 MoE/LoRA/all-reduce),支撑 RL 零 log-prob 偏差。

## 提案动机
`VLLM_BATCH_INVARIANT=1` 在 ROCm 已可用(#52231),但 Kimi-K3.1(hybrid KDA 线性注意力 + DSA sparse MLA、MXFP4 路由专家、LoRA)在 MI355X 上仅靠它仍非 batch-invariant。RL 中该模式保证 rollout token log-probs 与 trainer 完全一致("true on-policy");NVIDIA GB300 上已有 zero-mismatch RL overlay 方案,本 issue 跟踪 ROCm 对应物,按小 PR 逐个落地。跟踪父 issue #27433。经全模型 batch-size 扫描二分定位,破坏点包括:KDA decode/prefill 走不同内核、FLA 按形状 autotune、CUDA-graph replay 后 state slot 可能写两次;DSA indexer decode/prefill logits 不同、top-k tie order;MXFP4 MoE 按 token 数选 tile/split;TP all-reduce 随消息大小换算法;EP combine 按到达顺序求和;LoRA 两次 BF16 舍入等。

## 提出的方案
每项一个 PR,全部 gated 于 `VLLM_BATCH_INVARIANT`,默认路径不变:gfx950 BI matmul tile 表(小 M 分桶,BLOCK_K 保持 64,逐项与默认配置位等);LoRA 单次舍入 expand epilogue;DSA/indexer BI(decode 与 prefill 同法、确定性 top-k、BI 模式禁用 FlyDSL);AttnRes BI;KDA BI(pinned FLA `chunk_kda` 配置、decode 与 chunk 位等、CUDA-graph 可捕获、state-slot 写去重);BI 两段式 custom all-reduce;BI MXFP4 路由专家 MoE(向 AITER 提议 opt-in 确定性模式);ordered EP combine;ROCm Kimi-K3.1 的 batch-size sweep + 全模型 gate 测试。

## 优化点与收益
证据(2 节点 × 8 MI355X,vLLM rollout + Megatron trainer,LoRA r32):GRPO 每步 `rollout_probs_diff_max == 0`——30-step run、3,840/3,840 samples、11,893,214 tokens、max abs diff 0(逐 token 检查)。BI 代价(NVIDIA 公布的 Kimi-K3.1 RL shape:256 samples/step、2048 prompt/6144 response、DAPO-Math-17k):rollout +1.0%、整个 RL step −2.9% vs BI off。

## 现状与落地
open,提案内 checklist 均未勾选;代码目前为 out-of-tree 插件+补丁(verl draft PR verl-project/verl#8574),正就 MoE kernel 放 AITER 还是 vLLM、KDA BI 相对 GDN 工作的结构征求维护者意见。

## 关联
#52231、#27433、#65278、#42960/#48613、verl-project/verl#8574

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
