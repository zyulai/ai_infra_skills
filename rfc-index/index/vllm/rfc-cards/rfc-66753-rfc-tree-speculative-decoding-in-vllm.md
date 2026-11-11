# RFC #66753: [RFC]: Tree speculative decoding in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66753](https://github.com/vllm-project/vllm/issues/66753) |
| 作者 | chenaoxuan |
| 创建日期 | 2026-11-10 |
| 状态 | 🟢 open |
| labels | RFC, speculative-decoding |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
细化 vLLM 树形投机解码的调度、验证与提交全流程,让每步提出的 draft tree 在一次 target 验证中接受更多 token。

## 提案动机
树形 drafting 已成为投机解码的重要方向:每步提出多个续写,draft tree 可增加单次 target 验证通过的 token 数(文中引 PCTree、DARTree、PRESTO 三篇工作)。本 RFC 基于 vLLM RFC #65813,并借鉴 Ascend 上的原型实现(vLLM-Ascend RFC #18797),具体细化 vLLM 应如何调度、验证与提交 draft trees。

## 提出的方案
按五阶段执行流组织:
- Propose:生成有界(bounded)候选续写树,发布展平的候选 token ID,并保留拓扑及其与对应 worker 请求状态的关联。
- Schedule and prepare inputs:为候选行及其 KV 槽位记账;把树裁剪到预算内时保留祖先闭合(ancestor-closed)子集;展平候选与拓扑始终绑定于同一请求与同一 draft step。
- Verify:按深度分配 position ID,按节点分配互不相同的临时 KV 槽;每个 query 可看到已提交前缀及其自身在树中的路径(含自身节点)。
- Select a path:从根开始,把 target 选出的 token 逐层匹配到 child;遇到匹配缺失或到达叶子即停止,发出 target token,并遵循常规 stop 与长度处理。
- Commit:保留根节点,按路径顺序提交被接受的节点,更新请求状态,并按同一顺序收集下一轮 draft 所需的 target hidden states。

## 优化点与收益
正文未给出量化数字。机制层面的收益是:通过每步提出多个续写构成的 draft tree,一次 target 验证可接受的 token 数得以增加;具体验证/提交流程由上述五阶段执行流定义。

## 现状与落地
仍为 open,处于设计讨论阶段:正文给出动机与五阶段执行流设计,CC 了 @benchislett @ZJY0516 @TheEpicDolphin @drslark;正文未包含反馈期安排或实现时间表,落地进展需关注后续讨论与关联 RFC #65813。

## 关联
- vLLM RFC #65813(树形投机解码框架)
- vLLM-Ascend RFC #18797(Ascend 原型)
- PCTree(arXiv 2608.02123)
- DARTree(arXiv 2608.13524)
- PRESTO(arXiv 2607.22634)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
