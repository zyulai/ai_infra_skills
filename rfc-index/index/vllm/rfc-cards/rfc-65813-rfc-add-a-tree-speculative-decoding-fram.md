# RFC #65813: [RFC]: Add a Tree Speculative Decoding Framework

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65813](https://github.com/vllm-project/vllm/issues/65813) |
| 作者 | drslark |
| 创建日期 | 2026-11-05 |
| 状态 | 🟢 open |
| labels | speculative-decoding |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
提议在上游 vLLM 建立统一、可复用的树式投机解码框架:先定 tree 接口,再逐步支持各 attention backend 与动态树验证。

## 提案动机
树式投机解码是活跃研究方向,EAGLE-4、SpecPilot 等近期工作在探索树式 drafting 与验证。通过覆盖多个候选延续,树方法可以显著提升 draft acceptance,并增加每次 target verification 接受的 token 数。然而上游 vLLM 缺少统一、可复用的树投机框架,使这些方法难以实现与评估。

## 提出的方案
增量构建树式投机解码支持,分三步:
1. 在 model runner 与 attention layers 引入树接口:为 tree proposal、verification、accepted-path commitment(含 KV cache 与 drafter state 更新)定义公共 tree metadata 与通用接口。
2. attention backend 逐步支持:从静态树开始逐个 backend 启用 tree attention,每个 backend 显式声明其支持的能力。
3. 动态树验证:支持跨请求、跨解码步变化的树拓扑与节点数,实现自适应扩展与剪枝。
共享接口建立后,第 2、3 步可并行推进(正文附 mermaid 流程图:Step 1 → Step 2/Step 3)。现有 chain speculative decoding 在各阶段保持兼容。

验证计划:验证 attention isolation、verification correctness 与 cache consistency;将端到端时延、吞吐、每次验证接受的 token 数与内存开销,与 autoregressive 及 chain 投机解码对比。

## 优化点与收益
定性收益:树方法通过覆盖多个候选延续,可大幅提升 draft acceptance 与每次验证接受的 token 数;正文未给出量化数字。

## 现状与落地
仍 open,2026-11-05 刚创建,处于征集反馈的提案阶段。

## 关联
- EAGLE-4:https://github.com/eagle-hq/EAGLE-4
- SpecPilot:https://github.com/StargazerX0/SpecPilot

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
