# RFC #7351: Create speculative decode dynamic parallel strategy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7351](https://github.com/vllm-project/vllm/issues/7351) |
| 作者 | vladislavkruglikov |
| 创建日期 | 2024-08-09 |
| 状态 | ⚪ closed(关闭于 2024-08-15) |
| labels | RFC |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
提议为 vLLM 增加一套投机解码动态并行策略,由 5 个子特性(状态保存、prompt 随机 draft、多 token 返回、并行 scorer、组合策略)组成。

## 提案动机
正文动机仅一句:「Create new speculative decode dynamic parallel strategy for our team needs」——为团队需求创建新的投机解码动态并行策略;未展开问题背景或量化痛点(正文较简短,卡片仅覆盖其字面内容)。

## 提出的方案
5 个 feature,各由独立 PR 承载(每个 feature 的高层描述随 request for change 给出,更多细节在其 PR 内):
- #7358 保存投机解码状态:允许用户可选地接收 draft token 索引历史等投机解码产物(每个 speculative decode 算法步骤的记录)。
- #7359 从 prompt 随机采 token 创建 draft:实现把「从当前 prompt 随机采样的 token」并入先前 draft 的投机 proposer。
- #7361 允许 model executor 返回多个 next token:现有实现 decode 阶段只产出一个 last next token;该特性让内部 model runner 能返回一段 token 范围的 next tokens。
- #7362 并行 scorer:利用 #7361 的能力,在单次 forward 请求中运行 target model 并返回 next tokens 的分数。
- #7363 组合上述特性,形成用户可用的投机解码动态并行策略。

## 优化点与收益
正文未给出量化数字,也未描述预期收益幅度。

## 现状与落地
已关闭(2024-08-15,距创建仅 6 天)。正文未说明关闭原因,也未说明各子 PR 是否合入;落地情况需查关联 PR(#7358/#7359/#7361/#7362/#7363)。

## 关联
- #7358、#7359、#7361、#7362、#7363(5 个子特性 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
