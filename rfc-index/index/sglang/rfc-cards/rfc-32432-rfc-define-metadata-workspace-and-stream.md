# RFC #32432: [RFC] Define Metadata, Workspace, and Stream-Ownership Contracts for Dynamic CUDA Graph Replay

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32432](https://github.com/sgl-project/sglang/issues/32432) |
| 作者 | Phoenix3334 |
| 创建日期 | 2026-07-26 |
| 状态 | ⚪ closed(关闭于 2026-10-10) |
| labels | inactive |
| 热度 | 👍 0 · 💬 11 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
为动态 CUDA Graph capture/replay 定义九类显式运行时契约,把异步复用 GPU 状态的隐式假设变成可审查、可测试、可执行的规则。

## 提案动机
源自 DSpark compact ragged target-verify 调查(#31023 / #33356)中反复出现的 CUDA illegal-memory / graph 失败。调查把这些表面相似的失败分解为四层:control-plane consistency、kernel 产出 plan 的正确性、capture 初始化顺序、replay 期的 ownership/stream/lifetime 契约——不再混为单一的"metadata/workspace 生命周期"问题;同一 warmup-to-capture 不变量在 Full 与 Breakable 两个 graph backend 中独立出现,更说明契约应显式化并酌情共享执行。RFC 不宣称所有 DSpark 失败同根同因,只主张让异步 capture/replay 的假设可审查、可测试、可执行。(pack 正文在约 2 万字符处截断,尾部"Relationship to existing work"一节不完整。)

## 提出的方案
九类运行时契约,各附核心不变量:
1) 分布式控制面一致性——同一步各 rank 的布局决策(verify-token budget、graph tier、collective 形状等)必须一致;2) Graph-static metadata——replay 前 every dynamic value 已完全初始化、对消费流可见、存储覆盖完整 replay 生命周期;3) workspace ownership——最后一个 GPU 消费者完成前不得覆写/重绑/重分配;4) stream-ordering——跨流依赖用显式 event,不从 host 执行顺序推断;5) capture initialization——非 capture-safe 的 warmup/JIT/惰性初始化必须在 capture 前完成,需协同 capture 的 rank 在同一边界会合;6) graph/cache-key——区分 capture-shape 维度、容量维度、每 replay 可变内容,省略字段须说明为何共享安全;7) request/KV-slot 生命周期——CPU 逻辑完成不足以复用,须确认 GPU 消费完成;8) kernel 产出的 plan/index——断言结构性不变量(如 ragged_id < 真实行数),固定 launch geometry 等隐含假设要写成显式断言;9) debuggability——无同步、opt-in 的追踪,避免观察者效应抑制时序敏感故障。
采用分三阶段:文档+断言+回归 → 局部最小修复 → 提取跨后端共享 helper(候选:#33795/#34286 的 capture-initialization 边界);并明确 non-goals(不引入无条件热路径 synchronize、不做万能 workspace 等)。

## 优化点与收益
正文未给出整体量化收益,但记录了各层修复的验证数字(条件均为文中所述复现环境):
- #31195 广播 TP0 的最终 verify budget 后,TP8 测试矩阵 120/120 通过,verify_token_budget / sum(verify_lens) / target Graph-key divergence 均为 0。
- #32467 B30Z 单卡回归:unpatched 2592 bad plans / 6000 calls;patched 0 bad plans / 30000 calls。
- 审阅者去初始化简化实验(每 case 10,000 calls、每变体 30,000):原始版 13111/30000 OOB plans,#32467 版 0/30000,去初始化变体 0/30000。
- 全模型 B300/TP8:startup capture 3/3 READY(每形状 35/35)、Gate 1 320/320、强路径 512/512,无 GPU fault。
- B300/TP8 复现器上 #33795 only 仅 1/6 READY,对照 #32467 only 3/3 READY。

## 现状与落地
已关闭(2026-10-10,label inactive)。文中列出的四层问题各有对应 PR(#31195、#32467、#33795、#34286)推进修复;本 RFC 自身的共享契约/共用 helper 落地情况需查关联 PR。

## 关联
- #31023(DSpark 调查源头)
- #31195(跨 TP verify-budget 一致性)
- #32467(ragged prefill plan 竞态)
- #33795(FullCudaGraphBackend capture 边界)
- #34286(BreakableCudaGraphBackend 同一边界)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
