# RFC #10818: [RFC]: Disaggregated prefilling and KV cache transfer roadmap

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#10818](https://github.com/vllm-project/vllm/issues/10818) |
| 作者 | KuntaiDu |
| 创建日期 | 2024-12-02 |
| 状态 | ⚪ closed (2025-08-12) |
| labels | RFC, stale |
| 热度 | 👍 29 · 💬 23 |
| 分类 | CI / Infra / Process |

## 一句话结论
分离式 prefill(XpYd)与通用 KV cache 传输的社区路线图,按连接、兼容性、异步传输、编排等方向逐项推进。

## 提案动机
作者发布该 issue 作为分离式 prefill(以及通用 KV cache 传输)的正式路线图,邀请社区认领与贡献;正文自述 "Here is the roadmap for disaggregated prefill (and general-purpose kv cache transfer). Feel free to contribute."

## 提出的方案
路线图按九个方向组织,要点包括:
- XpYd 支持(X 个 prefill 实例、Y 个 decode 实例,两者 tp/pp 可不同):放弃 roi 拆维与直连多连接两个方案,转向"Xp 连接一个 KV cache server、再由该 server 连接 Yd"的 KVCache-store 式设计;
- 建立连接:定期发 dummy 请求保活、`vllm connect` 建连、跨节点连接 prefiller/decoder、直连 Engine 而非 API server 以提升性能;
- 兼容性:与 chunked prefill、prefix caching、pipeline parallel、多模态兼容;
- 异步 KV cache 传输:KV cache 预取、layer-by-layer 流水线;
- 内存控制:复用 vLLM page table 避免碎片、减少 tensor 拷贝(已推迟到 2025 Q2);
- 自适应与容错:批次中部分 KV cache 未到时只对缺失 token 做 prefill、允许 prefill/decode worker 相互转换;
- 第三方引擎集成:Mooncake、LMCache 已完成;InfiniteStore、Valkey 因开发者无响应被划掉;
- 持久化 prefix caching:支持取回前缀/中间连续段 KV cache 再补算剩余 token;
- 编排与测试:集中式 orchestrator 管理 worker 池、动态增删 worker、初始路由;离线分离式 prefill 测试已完成。

## 优化点与收益
预期收益集中在:异步/流水线化 KV cache 传输(预取、layer-by-layer)、减少内存碎片与 tensor 拷贝、按需仅补算缺失 token 等性能方向;以及 XpYd 弹性部署与容错带来的吞吐与可用性提升。正文未给出量化数字。

## 现状与落地
已关闭(2025-08-12):路线图中 Mooncake、LMCache 集成与离线测试等条目已勾选完成,多个方向(内存控制)被划掉或推迟,其余大量条目在 issue 关闭时仍未勾选;整体落地情况需查各条目关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/12957 (KVCache-store 式 XpYd / worker 转换)
- https://github.com/vllm-project/vllm/issues/11791 (`vllm connect` 建连)
- https://github.com/vllm-project/vllm/issues/10884 (Mooncake 集成)
- https://github.com/vllm-project/vllm/issues/12953 (LMCache 集成)
- https://github.com/vllm-project/vllm/issues/12418 (离线分离式 prefill 测试)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
