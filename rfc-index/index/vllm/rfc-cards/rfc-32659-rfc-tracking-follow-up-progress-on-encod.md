# RFC #32659: [RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#32659](https://github.com/vllm-project/vllm/issues/32659) |
| 作者 | fake0fan |
| 创建日期 | 2026-01-20 |
| 状态 | ⚪ closed(关闭于 2026-08-30) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Disaggregated Serving |

## 一句话结论
EPD(Encode–Prefill–Decode)分离合入后的后续优化跟踪清单,汇总真实场景反馈并逐项跟进解决。

## 提案动机
EPD 代码 #25233 已合入一段时间,期间社区收到了反馈,并在真实场景中遇到了一些问题。作者希望通过 RFC 把这些问题汇总跟踪,并按清单持续优化和解决。

## 提出的方案
以任务清单形式跟踪 6 项后续工作(勾选状态按正文原样):
- [x] Load encoder-only option(#30242)
- [ ] 优化现有 EPD proxy 实现(#31017)
- [x] 优化 scheduler 中的远程缓存检查(PR #32585)
- [ ] [EPD] 支持 EC cache deallocation(fake0fan/vllm#18)
- [ ] MoonCake ECConnector(#30468)
- [ ] Nixl ECConnector(fake0fan/vllm#20)

正文除此之外未给出各项的具体设计细节。

## 优化点与收益
正文未给出量化数字。收益为工程跟踪类:集中记录 EPD 分离在真实场景中的问题(EPD proxy 性能、EC 缓存释放、MoonCake/Nixl 等 EC 连接器支持),便于社区按项认领与推进。

## 现状与落地
已关闭(关闭于 2026-08-30,labels 含 stale)。关闭时清单 6 项中 2 项已完成(Load encoder-only option、scheduler 远程缓存检查优化),其余 4 项仍未勾选;各项最终落地情况需查关联 PR。

## 关联
- [#25233](https://github.com/vllm-project/vllm/issues/25233)(EPD 代码,已合并)
- #30242(Load encoder-only option)
- #31017(EPD proxy 优化)
- [#32585](https://github.com/vllm-project/vllm/pull/32585)(远程缓存检查优化,已合并)
- #30468(MoonCake ECConnector)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
