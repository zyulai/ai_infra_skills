# RFC #38175: [RFC]: Support ViT Full CUDA Graph (Tracker)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38175](https://github.com/vllm-project/vllm/issues/38175) |
| 作者 | shen-shanshan |
| 创建日期 | 2026-03-26 |
| 状态 | 🟢 open |
| labels | RFC, help wanted, kimi, multi-modality |
| 热度 | 👍 2 · 💬 49 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
为多模态模型的 ViT 视觉编码器启用完整 CUDA Graph 捕获与重放,消除逐 kernel 启动开销;本 issue 为集成进度跟踪表。

## 提案动机
Qwen3-VL、Qwen3.5、GLM-V、Kimi K2.5 等多模态大模型依赖 ViT 编码器在送入语言模型主干前处理视觉输入。生产服务中,ViT 前向要启动大量小 CUDA kernel(patch embedding、layer normalization、多头自注意力、MLP 投影等),每个都在主机侧产生可观的 kernel 启动开销。目前 vLLM 只对模型 decoder(LLM)部分做 CUDA graph 捕获——已证明能有效降低启动开销、提升吞吐——而 ViT 编码器仍以 eager 方式执行,每次前向都从头重新启动所有 kernel。把完整 CUDA graph 支持扩展到 ViT 编码器,可将整个编码器前向捕获为单一图并重放,消除逐 kernel 启动开销,为多模态模型带来更一致的低延迟推理。

## 提出的方案
以 tracker 形式跟踪四类工作条目:
- 模型集成(27 个 PR 条目,绝大多数已勾选):按参考实现 Qwen3-VL 为各多模态模型实现 ViT CUDA graph 接口;
- Bugfix / 改进(17 个 PR 条目,仅 1 项未完成);
- 测试覆盖(2 个 PR,均完成)与文档(4 个 PR,均完成);
- 固化集成工作流:1) 参照 Qwen3-VL 为模型实现 ViT CUDA graph 接口;2) 完成 ut/e2e/benchmark 等测试;3) 更新文档中的受支持模型列表(docs: cuda_graphs_multimodal);4) 将模型加入 CI 测试。

## 优化点与收益
收益为消除 ViT 编码器逐 kernel 的主机侧启动开销,使整个编码器前向以单图重放执行,降低多模态推理延迟并使其更稳定。正文未给出量化数字(如具体延迟或吞吐提升百分比)。

## 现状与落地
state=open:该 tracker 正在活跃推进——模型集成、bugfix、测试、文档中已有大量 PR 勾选完成,仍有数个模型集成 PR(如 #64311、#44394、#56277、#56820、#57379、#63074、#63374)与 1 个 bugfix PR(#65351)未完成;labels 含 help wanted,持续征集贡献者。

## 关联
- https://github.com/vllm-project/vllm/pull/35963 (首个模型集成 PR @b-mu)
- https://github.com/vllm-project/vllm/pull/41736 (模型集成 @johncalesp)
- https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/#model-integration-via-supportsencodercudagraph (受支持模型列表文档)
- https://github.com/vllm-project/vllm/pull/37914 (文档 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
