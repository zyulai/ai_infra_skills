# RFC #66418: [RFC]: DBO support for the DeepEP v2 backend in Model Runner V2

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66418](https://github.com/vllm-project/vllm/issues/66418) |
| 作者 | algebra-MCX |
| 创建日期 | 2026-11-08 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |

## 一句话结论
提议在 Model Runner V2 上为 `deepep_v2` all2all 后端加入 DBO 支持,避免 CUDA 侧支持 DBO 的后端只剩 `nixl_ep`。

## 提案动机
`deepep_v2` 已合入 main,但 `--enable-dbo` 目前无法与其组合。跟踪问题 #62033 计划退役 `deepep_low_latency` 与 `deepep_high_throughput`;若退役先发生,`nixl_ep` 将成为唯一支持 DBO 的 CUDA 后端,此 issue 就是补上这个缺口。此前的原型 #52177 基于 Model Runner V1 编写,自 8 月起无人评审,作者提议在 MRV2 上重做而非继续 rebase。另指出当前 `deepep_v2.py` 已按微批保留 handle 并在 DBO 下强制同步 combine,但没有 `dbo_yield_*`/`dbo_switch_*` 调用,也未实现 `supports_dbo()`。

## 提出的方案
首个切片范围:单节点、DP/EP、CUDA、MRV2 上的 eager DBO;FULL CUDA Graph DBO 与多节点放在其后。
- 执行模型:把 #52177 的 yield/switch 调度移植到当前 `deepep_v2.py` —— 在计算流捕获前一事件、让位给另一微批、在 comm 流执行 dispatch/combine 后切回;DBO 下维持现有同步 combine;实现 `supports_dbo()`,以下步骤通过后才把 `deepep_v2` 加入 CUDA allow-list。
- MRV2 集成:接入 MRV2 微批路径而非 #52177 的 V1 runner 改动,并核查哪些 V1 改动(如 `allow_microbatching=False` 的 profile run)在 MRV2 中仍然需要。
- 缓冲与事件生命周期:确保一个微批的 dispatch/combine 缓冲与事件在另一微批仍在读取时不会被复用,从 #60735 对 v1 high-throughput 缓冲同类问题的修法出发。
- CUDA Graph:检查两微批下的图捕获;若首个切片在 capture 下不安全,则该后端保持 eager DBO 并在 config 检查中给出明确报错。
- 测试与基准:覆盖按微批句柄索引、`supports_dbo`、allow-list 的单测,以及单节点 DBO 开/关的端到端 A/B 与 trace。

范围外:DeepEP v2.5 迁移(ElasticBuffer → EPBuffer,#62033 跟踪)、改动 DeepEP kernels、SBO(#64776)。

## 优化点与收益
正文未给出量化数字。收益机制是让本微批的 dispatch/combine 与另一微批的计算重叠:完成标准要求 `vllm bench serve` 的吞吐/TPOT 在 DBO 开/关下对比并附 trace(显示 dispatch/combine 与另一微批计算重叠),且一个 MoE 模型在 DBO 开/关下输出/logprobs 一致、uneven microbatches 反复执行不挂起。

## 现状与落地
仍为 open,处于征求维护者意见阶段:作者列出三个问题 —— 是否可在新 MRV2 PR 中基于 #52177(会注明并链接原原型,或改由 @SageMoore 继续、本人协助评审/测试)、是否应在 #62033 退役 v1 DeepEP 后端之前落地以免 CUDA DBO 一度只剩单后端、是否仅支持 MRV2 即可。Feedback Period 为两周,CC @SageMoore @tlrmchlsmth。

## 关联
- #62033(DeepEP 后端退役 tracker)
- #52177(V1 原型 PR)
- #52781(v2 async finalize)
- #60735(v1 high-throughput 缓冲生命周期修复)
- #64776(SBO)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
