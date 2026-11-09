# RFC #53916: [RFC]: Native Text Watermarking Support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53916](https://github.com/vllm-project/vllm/issues/53916) |
| 作者 | TQCB |
| 创建日期 | 2026-08-26 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 17 · 💬 8 |
| 分类 | Feature / Ecosystem |

## 一句话结论
提议在 vLLM 采样过程中原生支持文本水印(Gumbel-max 起步),满足合规溯源需求并附带可复用的检测原语。

## 提案动机
水印已是负责任 LLM 部署的重要部分,并被法规要求支持溯源与透明(如 EU AI Act Article 50(2)、California SB 942);它是对采样过程的修改,下游消费者无法自行实现。也不能作为 out-of-tree 功能(如 CustomLogitsProcessor):部分方案(如 gumbel-max)需修改采样本身、需要访问最终输出分布、生产级实现还需支持投机解码并调整 rejection sampling。

## 提出的方案
- 原生水印在采样阶段实现,初始算法为 Gumbel-max:PRF 生成 keyed 均匀随机向量 r_k~U(0,1),每位置采样 argmax_x r_k(x)^(1/p(x)),不改变期望分布 p;检测时重算 r_t 并按 score = sum_t -log(1 - r_t) 打分,零假设下服从 Gamma(n, 1)
- 引擎启动配置 `WatermarkConfig`(algorithm/key/context_width);`SamplingParams` 增加 watermarking 字段支持请求级开关;greedy 请求不加水印
- 共享 `Watermarker` 接口在最终 token 选择处替换普通随机采样,与 model runner 解耦;初始支持 Model Runner V2,投机解码暂不在范围
- 检测原语:`GumbelWatermarkDetector` 接收 token IDs,输出 aggregate score、p-value、scored tokens 与阈值分类;含 repeated-context dedup(p 值校准所需)
- PRF 模块化系统:初始选择内部维护的 Philox(philox4x32-10-v1),保持稳定;HMAC 为潜在备选

## 优化点与收益
照抄数字(Qwen/Qwen3.5-2B、GSM8K、H100 三次取中位):吞吐 baseline 11,056.947 vs 水印 10,985.863 token/s/gpu(Delta -0.643%);预期吞吐损失 O(0.3% for 1B models, 0.1% for 30B+ models)(Gumbel-max)、O(1% - 0.5%)(SynthID-Text,引 TextSeal Table 4)。检测阈值 p<=0.01(1% FPR,8 keys 平均):水印响应 positive rate 42.13%(T=0.7)/84.17%(T=1.0),未水印 1.135%-1.172%。注意:该实现存在多样性灾难性坍缩,输出长度 T=1.0 时 +18.23 tokens(95% CI [+13.98, +22.50])。

## 现状与落地
open。草案实现见 PR #54053,作者建议将其视为 RFC 的技术支持;正在等待社区评审讨论。

## 关联
- 草案 PR:https://github.com/vllm-project/vllm/pull/54053
- Gumbel-max(Aaronson and Kirchner, 2023):https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf
- SynthID-Text:https://www.nature.com/articles/s41586-024-08025-4#MOESM1
- TextSeal:https://arxiv.org/pdf/2605.12456
- EU AI Act Article 50:https://artificialintelligenceact.eu/article/50/

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
