# RFC #17767: [Feature] [RFC] Extend EAGLE worker to support a step-disaggregated draft model, PRISM

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17767](https://github.com/sgl-project/sglang/issues/17767) |
| 作者 | Akemiiii |
| 创建日期 | 2026-01-26 |
| 状态 | ⚪ closed (2026-03-28) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
扩展 EAGLE worker 支持投机解码方法 PRISM:按 draft 步路由到不同 transformer 层,提准不加每步开销。

## 提案动机
LMSYS 博客与 META 技术报告表明,大训练语料与多层 draft 模型比原版 EAGLE-3 有更好的端到端表现(投机更准),但会增加 draft 模型推理开销与训练成本。作者正在实现的 PRISM 追求更好的折中:在不增加每 draft 步计算成本的前提下,提高 draft 模型准确率与接受长度——做法是把每个 draft 步路由到不同的 transformer layers(modules)。架构细节与作者在 SGLang 原型实现上的性能结果见 SpecForge issue,更多结果将在后续 preprint 给出。

## 提出的方案
- 新增一个 SGLang flag 以启用 PRISM。
- 扩展现有 EAGLE worker 支持 PRISM。
- 在 python/sglang/srt/models 新增 PRISM 模型。
- 在模型的 config.json 中增加键值对,指定 PRISM 中 draft step 到 module 的映射。
- 修改 eagle_draft_cuda_graph_runner.py 与 eagle_draft_extend_cuda_graph_runner.py。

## 优化点与收益
正文未给出量化数字(性能结果放在 SpecForge issue 的原型实现中,preprint 待发布)。

## 现状与落地
已关闭(2026-03-28)。正文以提问形式征求维护者指导:应基于 V1 还是 V2 版 EAGLE worker 实现;继续用 benchmark/mtbench 脚本还是换 python3 -m sglang.bench_serving 做基准;对集成路径有无建议。正文未写明是否落地,需查关联 PR。

## 关联
- LMSYS 博客: https://lmsys.org/blog/2025-12-23-spec-bundle-phase-1/
- META 技术报告: https://arxiv.org/abs/2508.08192
- SpecForge issue #444: https://github.com/sgl-project/SpecForge/issues/444

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
