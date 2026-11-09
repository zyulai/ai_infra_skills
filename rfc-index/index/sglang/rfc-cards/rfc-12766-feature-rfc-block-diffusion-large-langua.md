# RFC #12766: [Feature] RFC: Block Diffusion Large Language Model (dLLM) Framework In SGLang

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12766](https://github.com/sgl-project/sglang/issues/12766) |
| 作者 | ClawSeven |
| 创建日期 | 2025-11-06 |
| 状态 | ⚪ closed (2026-03-16) |
| labels | (none) |
| 热度 | 👍 42 · 💬 19 |
| 分类 | Other |

## 一句话结论
复用 SGLang 现有 Chunked-Prefill 管线支持 Block Diffusion 语言模型(dLLM),几乎不改核心框架。

## 提案动机
LLaDA(人大与蚂蚁合作的首个扩散语言模型)展示出扩散范式在数据理解与推理速度上的优势,但把 DLM 规模化时遇到评测性能与 RL 后训练支持等工程挑战。现有扩散推理工具(如 fast-dllm)偏研究用途,难以支撑 batching、调度、RL 生态集成与并行等生产需求;而 SGLang 目前只支持自回归计算范式,尚未适配扩散计算。挑战是:如何在不损害现有架构的前提下,让 DLM 享受 SGLang 全部优化。

## 提出的方案
- 核心洞察:双向注意力扩散模型因 KV Cache 利用率低正转向 Block Diffusion 架构,其计算模式与 SGLang 的 Chunked-Prefill 高度相似。
- 复用 Chunked-Prefill 管线,改动集中在 prefill adder 与 chunked reqs,使单次计算周期能处理多个 Diffusion Block(如 LLaDA-2 的 Block Size 为 32 token),解决 batching 问题;原 generate request 执行流不变。
- 在 TP Worker 与 Model Runner 之间插入扩散算法抽象层:Worker 识别扩散模型后调用算法的 run 函数,内部用前向迭代循环驱动 Model Runner,直到整个 Block(如全部 32 token)解码完成。
- Attention 差异:Block Diffusion 用 Block-wise Causal Mask(块内双向注意力,几何上是矩形 mask),对历史 KV 的 Cache Query 部分与 Chunk Prefill 完全相同。
- KV Cache:块内 32 次迭代不把新 KV 写入 Request Token Pool,整块 unmask 后由一次完整前向把该块 KV 一次性写入。
- 动态 batching:按 max-running-requests 定 batch size,用 remaining total tokens 控制实际运行请求数;到达 EOS 的请求在 block diffusion batch 完成后移除,有剩余 Token Budget 且有等待请求时补入下一轮。

## 优化点与收益
正文未给出量化数字。正文列出已实现的关键特性:Block Diffusion 框架主逻辑、完整 KV Cache 支持、LLaDA-2.0-mini/flash 集成、自定义解码策略(如 low-confidence policy)、Streaming I/O、batching、TP、CUDA Graph。

## 现状与落地
已关闭(2026-03-16)。正文 Work items 显示上述关键特性与多数 TODO 已勾选实现;长期路线(overlap 调度、扩散投机解码、兼容非 block 扩散模型如 LLaDA/RND1)仍未勾选。落地情况需查关联 PR。

## 关联
- LLaDA 论文: https://arxiv.org/abs/2502.09992
- LLaDA2.0-flash-preview: https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview
- LLaDA2.0-mini-preview: https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
