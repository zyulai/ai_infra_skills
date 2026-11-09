# RFC #62150: [RFC]: PD disaggregation + RL lifecycle (pause, sleep, abort) tracking

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62150](https://github.com/vllm-project/vllm/issues/62150) |
| 作者 | aoshen02 |
| 创建日期 | 2026-10-17 |
| 状态 | 🟢 open |
| labels | RFC, kv-connector |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
追踪 PD 分离 + RL 场景下 lifecycle(pause/sleep/abort)修复的合并顺序,目标是让 MooncakeConnector 生命周期矩阵从 8/27 达到 27/27 通过。

## 提案动机
在 PD 分离(disaggregation)上做 RL 训练时,每次权重更新前后都要 pause、sleep、abort 并 reset cache。在 main(`ce49c47`)上,MooncakeConnector 在 GB300 上 27 个 lifecycle 用例只有 8 个通过;其余用例要么让 pause 阻塞 480 s,要么卡死 D 一侧,要么返回 500。(正文较简短,卡片仅覆盖其字面内容;附有 tracker 链接。)

## 提出的方案
按以下顺序合并(以 checkbox 追踪进度):
- [x] #59625 Mooncake sleep mode
- [x] #60572 在 wait pause 中排空(drain)async KV 加载
- [ ] aoshen02/vllm#86 及时应答每个 pull 并立即释放被丢弃的 transfer(R1–R3)
- [ ] aoshen02/vllm#87 在 cache reset 前释放 transfer 持有的 KV(R4)
- [ ] aoshen02/vllm#88 prefill 失败时停止对应的 decode 流(R5)
- [ ] aoshen02/vllm#104 让 P 重建被 cache reset 丢弃的 KV(可选,类似 SGLang 的 rebootstrap)
fork PR 会逐个开出;全部合入后,矩阵 27/27 通过。

## 优化点与收益
照抄数字:8 of 27 → 27 of 27 lifecycle cases pass(MooncakeConnector on GB300);消除 pause 阻塞 480 s、D 卡死与返回 500 等失效模式。

## 现状与落地
仍 open,作为进行中的 tracker:6 个条目中前 2 项已完成,其余 4 项 fork PR 待逐个开出。

## 关联
- #59625(Mooncake sleep mode)
- #60572(drain async KV loads)
- #59055(sleep-mode memory)
- Tracker:https://claude.ai/artifact/5yU9HihNDSW86B1oZFL2mc

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
