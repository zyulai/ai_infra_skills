# RFC #16546: [RFC] SGLang-Omni Design

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16546](https://github.com/sgl-project/sglang/issues/16546) |
| 作者 | FrankLeeeee |
| 创建日期 | 2026-01-06 |
| 状态 | ⚪ closed(关闭于 2026-05-09) |
| labels | (none) |
| 热度 | 👍 54 · 💬 9 |
| 分类 | Other |

## 一句话结论
为多模态/Omni 模型设计分阶段服务框架:stage 化拆分、灵活部署与复杂数据流编排。

## 提案动机
开源与闭源模型都在变得多模态化:Gemini 可输入文本、图像、视频、音频并输出文本、图像、音频,开源的 Qwen-Omni 也能做到类似的事;多家科技巨头的研究者在公开演讲中都预期 2026 年会出现更多 omni 模型。SGLang 团队因此认为需要引入新模块来承接。Omni 模型组件远多于 LLM(如 Qwen2.5-Omni 有 6 个组件:2 个 encoder、thinker、talker、codec decoder),数据流路径也多(Qwen2.5-Omni 有 8 种输入输出组合),组件如何放置、如何调度成为系统设计的突出难点。

## 提出的方案
- Inter-/Intra-Disaggregation:Inter 把整个模型拆成多个 stage,各 stage 有独立的调度与执行逻辑,tensor 经 Mooncake 或共享内存通信;Intra 是 stage 内多进程拆分(如 PD disaggregation),不由 SGLang-Omni 直接控制,复用现有实现。
- Multi-Scheduling:每个 stage 可采用各自调度策略,如 continuous batching、static grouping。
- Multi-Path:针对 Early End、Cyclic Flow、Multiple Receivers 三类数据流,通过 object message queue 指定目的地(跨 stage 直连、回传上一 stage、多接收者并行分发)。
- Multi-instance:慢 stage 水平扩容多实例(如 stage 1 每秒处理 4 个样本、stage 2 仅 2 个时扩 stage 2),用 SGLang Router 做负载均衡并按负载特征选调度策略。
- 以 YAML 定义 stage 数据流(scheduling_policy、disaggregated、out_flow、host、port、devices 等),借鉴 torchtitan 支持命令行覆盖配置;API 设计以图示给出。

## 优化点与收益
正文未给出量化数字。正文的量化内容仅为说明多实例必要性的示例(stage 1 每秒 4 样本 vs stage 2 每秒 2 样本),非实测收益。

## 现状与落地
已关闭,落地情况需查关联 PR。正文为设计提案,未写明实现进展。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
