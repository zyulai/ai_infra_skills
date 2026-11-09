# RFC #52038: [RFC]: LoRA adapter support for DFlash speculative decoding draft models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52038](https://github.com/vllm-project/vllm/issues/52038) |
| 作者 | anshkaggarwal22 |
| 创建日期 | 2026-08-12 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
让 DFlash 投机解码草稿模型支持 LoRA adapter:用一个基础 drafter 加小适配器替代每域全量 drafter,大幅省内存与加载开销。

## 提案动机
vLLM 已支持在单一基座 target 模型上服务多个 LoRA adapter,但投机解码仍要求每个领域一个完整训练的 draft model:服务多域时每个域都要一个 drafter(如 0.8B)常驻内存,并有相应的加载/切换 I/O 代价。作者提议把 LoRA adapter 应用到 DFlash drafter,让单一基础 drafter 经小 adapter 按域特化,而非每个域放一个完整 drafter。

## 提出的方案
允许 LoRA adapter 独立于 target 模型应用到 DFlash drafter:
- adapter 包裹除 `lm_head` 与 `embed_tokens` 之外的所有 drafter 层,从 stage-2 训练的 LoRA checkpoint 加载;
- drafter LoRA 通过 speculative config 中的新增参数指定,与任何 target-model adapter 区分开;
- 将正确的 adapter 路由到 drafter,并沿其 forward 路径应用。
初始支持以 DFlash drafter 为目标(实测 GPT-OSS-120B target + 0.8B DFlash drafter),区别于 target 侧多 adapter 工作(#49705、#48297)及此前 Eagle 式 draft-model 尝试。Prior art:已关闭 PR #11966(V0、Eagle 时代)曾因 draft 与 target 的 vocab padding 不同撞上 vocab-size mismatch(如 768 vs 4096),进而禁用 draft 侧 LoRA;作者在 DFlash 测试中未遇到该 mismatch——可能正因 adapter 排除了 `lm_head`/`embed_tokens` 这些 padded-vocab 层,是否对所有 DFlash 配置成立欢迎维护者确认。备选方案:全量 per-domain drafters(status quo,简单但昂贵);离线把 LoRA 合并进 drafter 权重(免运行时 adapter 逻辑,但每域仍存全量合并 drafter,失去 28 倍节省与按请求可切换性)。动态服务 adapter 可同时保留两者。

## 优化点与收益
照抄数字:R64 adapter 约比其替代的 0.8B drafter 小 ~28x;drafter 质量与全训 per-domain drafter 的差距保持在 ~2% 以内(measured by acceptance length behavior)。条件:GPT-OSS-120B target + 0.8B DFlash drafter 的实验;作者说明部分实验结果涉及 Capital One AI Foundations 任职期间的涉密数据与领域,只能分享部分。计划测试:base drafter / LoRA'd drafter / 全训 per-domain drafter 的 acceptance 行为与吞吐对比,以及 N 域下 adapter vs 全量 drafter 的内存占用对比。

## 现状与落地
仍 open,feedback period 一周,CC @LucasWilkinson @WoosukKwon;处于等待维护者反馈阶段。

## 关联
- #49705、#48297(target 侧多 adapter 工作)
- PR #11966(V0/Eagle 时代的 LoRA + spec decode 尝试)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
