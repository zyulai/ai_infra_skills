# RFC #50853: [RFC]: Complete Model Runner V2 pipeline parallelism: scheduling, speculative decoding, and PP-first DP/EP topology

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50853](https://github.com/vllm-project/vllm/issues/50853) |
| 作者 | LostFox11 |
| 创建日期 | 2026-08-03 |
| 状态 | 🟢 open |
| labels | k3, kimi |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Speculative Decoding |

## 一句话结论
补全 MRV2 流水线并行的功能组合:均衡准入、MTP/EAGLE3 等推测解码与 PP 协同、声明式 PP 传输与 PP-first 多机拓扑。

## 提案动机
MRV2 PP 已能重叠多批减少流水线气泡(#42187),但周边功能与 PP 组合不佳:突发请求可在一个 batch 占满全局请求槽、后续 PP batch 为空;last rank 本地应用的 MTP draft 需发布给非 last rank;EAGLE3/DFlash/DSpark 目前在 MRV2 中与 PP 显式互斥(辅助 hidden states 在前级产生、由 last rank 的 proposer 消费);PP 切分可能落在 index 生产层与复用层之间;DP-first 部署让 EP/TP 域跨节点而 PP 反而留在机内。共同主题:PP 还引入 forward state 与 deferred result state 两条方向、生命周期不同的状态传输路径,逐个加一次性通信代码会越来越脆弱。

## 提出的方案
四个可独立评审的 workstream:
- A 均衡请求准入:对 WAITING 新准入设软上限 `ceil(num_requests / pp_size)`(除数取 `pp_size` 而非 `pp_size + 1`),保留 work conservation;原型 #50410 在 pp_size=2、max_num_seqs=25、50 个短请求下把 `25, 0, 25, 0` 变为约 `13, 12`。
- B 声明式 PP 传输:forward payload(相邻 rank→相邻 rank)与 deferred result payload(last rank→各非 last rank)为两个独立 schema;`PPTensorSpec` 描述 plane、boundary_id、slot、carry、lifetime 等;可选字段关闭时不增加字节或 collective。
- C 推测解码组合:C1 MTP draft 反馈同步(#52295 实现窄范围部分),token 线载荷 `[B, (K + 1) + K] = [B, 2K + 1]` int64、不新增 collective,step `T` 发布、`T + pp_size` 应用;C2 EAGLE3/DFlash/DSpark 辅助 hidden states 走 forward 面、归一为全局 activation-boundary ID、`until_last` 生命周期;C3 跨 PP 切分的 index-sharing 状态,契约 TP-replicated `[N, index_topk]` `torch.int32`。
- D PP-first 多机放置:先把节点分配给 PP stage,EP/TP 通信留在节点内、PP 流量走节点间;不改变 DP 路由/虚拟前向等协议。

## 优化点与收益
正文未给出端到端性能数字 —— Workstream A 给出的是调度形状示例(上述 `25, 0, 25, 0` → 约 `13, 12`)而非提速声明,并要求验证涵盖 TPOT 与 proposer latency;目标收益为消除 PP batch 空转、解除 spec-decode 与 PP 的互斥、减少跨节点 EP 通信。

## 现状与落地
仍在讨论/已有原型:open 状态;Workstream A 原型在 PR #50410,C1 的窄同步部分已由 #52295 实现(正文强调其为传输/状态同步改动,不等于全部 MTP 模型使能)。注:pack 正文在 20k 字符处截断,Workstream D 的 D1 论证后半及之后内容未收录。

## 关联
- #42187(MRV2 PP 基础)
- #50410(Workstream A 原型)
- #52295(MTP draft 反馈同步)
- #46994(模型侧 PP capability 与同 rank buffer 修复)
- draft #47686(字段打包方向)
(以上正文均以编号引用,未附链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
