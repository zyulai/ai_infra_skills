# RFC #31483: [RFC]: a efficient adaptive rejection sampling for accelerating speculative decoding.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31483](https://github.com/vllm-project/vllm/issues/31483) |
| 作者 | sunchendd |
| 创建日期 | 2025-12-29 |
| 状态 | ⚪ closed(关闭于 2026-01-04) |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |

## 一句话结论
提议引入 EARS 自适应拒绝采样:按目标模型置信度动态放宽接受阈值,减少投机解码中的随机拒绝。

## 提案动机
投机解码核心的拒绝采样当前依赖固定、与上下文无关的随机阈值,导致明显的"随机拒绝"问题:高不确定性生成场景下,草稿模型提出的合理候选 token 常仅因接受比例略低于固定阈值而被随机拒绝,浪费了验证后续草稿 token 的算力,严重限制投机解码对创造性/开放性任务的加速潜力。作者观察到草稿接受率随生成不确定性升高而大幅下降——temperature=0 时超过 80%,temperature=1 时仅约 30%——其根源是固定阈值无法顾及目标模型预测在不同生成步的内在不确定性,需要一个感知模型置信度的自适应阈值机制。

## 提出的方案
采用 arXiv 论文 2512.13194 提出的 Efficient Adaptive Rejection Sampling(EARS),核心创新是根据目标模型的预测置信度动态调整接受阈值:
1. 定义不确定性:`Uncertainty = 1 - max(P_target)`,度量当前步模型的不确定程度;
2. 计算动态容忍度:`Tolerance = β * (1 - max(P_target))`,β 为可调超参数;
3. 修改接受条件:由传统规则 `P_target / P_draft ≥ U`(U 为均匀随机数)改为 `P_target / P_draft ≥ U - Tolerance`。
模型高度不确定(低 max(P_target)、高 Tolerance)时有效接受阈值降低,"赦免"本会被随机拒绝的合理候选;模型自信(高 max(P_target)、低 Tolerance)时标准保持严格,保住生成质量。无需修改模型架构,可借助预计算等优化以最小工程开销接入现有投机解码框架。

## 优化点与收益
照抄正文数字:
- 「Achieves up to ~18.12% higher token throughput in open-ended generation tasks」——开放生成任务 token 吞吐最高提升 ~18.12%;
- 「negligible accuracy drop (e.g., <=0.84% on GSM8K) in tasks requiring precision」——精度任务质量影响可忽略,GSM8K 上 accuracy 下降 ≤0.84%。
测试条件:正文仅给出任务类型(open-ended generation tasks、GSM8K),未注明具体模型与硬件,数字出自所引用的 arXiv 论文。

## 现状与落地
已关闭(关闭于 2026-01-04,创建约一周后)。正文的 Feedback Period、CC List、Any Other Things 均为 _No response_,无实现或讨论结论,落地情况需查关联 PR。

## 关联
- [arXiv 2512.13194](https://arxiv.org/abs/2512.13194)(EARS 方法论文)
- 正文未列出其他 RFC/PR/文档链接

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
