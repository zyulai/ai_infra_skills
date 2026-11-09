# RFC #54080: [RFC]: TreeWY: Tree Speculative Decoding for hybrid (GDN) models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54080](https://github.com/vllm-project/vllm/issues/54080) |
| 作者 | sneha5gsm |
| 创建日期 | 2026-08-27 |
| 状态 | 🟢 open |
| labels | RFC, speculative-decoding |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |

## 一句话结论
提议为 vLLM 的混合(GDN)模型支持树形投机解码,以 WY/UT 状态重建消除 recurrent-state 快照内存瓶颈。

## 提案动机
混合模型(Qwen 3.5 系列)投机解码存在 recurrent-state 内存瓶颈:`mamba_cache_mode="all"` 下 draft N 个位置需要每序列每层 N+1 个完整状态快照,Qwen3.5-397B 上为 90 MiB/seq,depth-3 链 360 MiB、[2,2,2] 树(14 节点)约 1.2 GiB,与 KV cache 争同一预算。作者改为 commit 时重建 accepted state(WY/UT 表示的 gated delta rule,一次严格下三角求解覆盖整个 verify 窗口),并扩展出树形投机解码。

## 提出的方案
- 支持基于 GDN 的 Qwen 混合模型的树形投机解码;已有实现分支(见关联)。
- 与 ReplaySSM 的差异仅一个谓词:作者实现加载显式祖先掩码,commit 变为祖先掩码归约而非游标移动,因此树形可达。
- 作者实测 ReplaySSM 在混合模型链式投机上更快、模型支持更广,愿与其团队协作补齐树形支持。

## 优化点与收益
Spec-Bench 上验收率随宽度单调上升(3.23 → 3.44),但各宽度下吞吐均下降:分支 verify 步目前无法被 CUDA graph 完整捕获,树形(宽度>1)需非因果祖先掩码导致回退 PIECEWISE。作者明言树形投机暂非吞吐收益(draft trees are not a throughput win yet)。

## 现状与落地
open,提案阶段,已有可运行实现分支,等待评审并寻求与 ReplaySSM 团队协作。

## 关联
论文 https://arxiv.org/abs/2608.20961;ReplaySSM https://github.com/vllm-project/vllm/issues/47572;实现 https://github.com/vllm-project/vllm/compare/main...sneha5gsm:vllm:treewy-gdn-spec-decode

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
