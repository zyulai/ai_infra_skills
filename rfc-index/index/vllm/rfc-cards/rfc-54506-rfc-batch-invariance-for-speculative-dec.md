# RFC #54506: [RFC]: Batch invariance for speculative decoding needs to cover the forward pass (M=1 vs M=k+1)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#54506](https://github.com/vllm-project/vllm/issues/54506) |
| 作者 | Leonccaa |
| 创建日期 | 2026-08-31 |
| 状态 | 🟢 open |
| labels | batch-invariance, quantization, speculative-decoding |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
论证投机解码的 batch invariance 必须覆盖前向计算(M=1 vs M=k+1),请求把它纳入 #52522 scope、作为 kernel 与编译层的一等契约。

## 提案动机
temperature 0 下启用投机解码会改变生成序列(ngram 案例见 #41758;#52522 只覆盖 sampling 层)。作者团队在 125B hybrid linear-attention/MoE 模型(原生 MTP head,TP4,FP16)上多日取证后认定主因是前向算术而非采样:verify pass 以 M=k+1 行计算、plain decode 以 M=1 计算,当前栈不保证两者 bitwise 一致;top-2 logit margin 处于 ULP 量级的决策点被 1-ULP 差异翻转后,贪心解码把翻转放大为永久分叉。实测症状:draft acceptance ~100% → 72%,decode ~110 → 89 tok/s。四类来源(均有 bitwise 证据):A kernel 级 M 相关 reduction(split-K、norm 按 M 选 1024/512 宽);B 编译/fusion 拓扑(Inductor 按形状特化生成 reduction、opaque custom op 充当 fusion barrier);C 引擎执行形式不对称(spec 配置 gate 把整个 attention-block family 移到 eager);D 配置激活成员(量化准入、prefetch vs mmap/lazy 加载从 token 0 改变轨迹)。逐个点修复不收敛:跨 successive 真修复,首个发散 token 序号 14 → 2 → 14 → 10 迁移。

## 提出的方案
作者明确不提设计,只要 scope 决策:把「同一逻辑行跨 M 的前向 batch invariance」当作 kernel 与编译层的 first-class contract(如同 `VLLM_BATCH_INVARIANT` 对 batch composition 的处理),扩展 #52522 使其覆盖 sampling 边界之外。待决问题:(1) 该问题属于 #52522 scope 还是独立工作流;(2) verify↔decode bitwise parity 应成为 batch-invariant 模式下文档化保证(承担相应成本),还是显式非保证(让用户不再把贪心输出变化当成自己配置的 bug);(3) 若为保证,是否必须覆盖编译层——Class B/C 表明只保证 kernel 不够,fusion 拓扑与每引擎执行形式本身就会改变结果。附录提供归因方法学(路由命中证明、真实 checkpoint 权重的 per-instance 算子 gate、观测中立性等)、各类实测签名与最小复现配方。

## 优化点与收益
照抄数字(hand-written F16 GEMM,M=5 vs 5×(M=1)、随机权重:~92–114 of 20480 元素差 1–2 ULP;同一 kernel 在真实 checkpoint 权重 140+ 实例 × M=2..8 上零表达;gated-RMSNorm fusion,M=1 vs M=2:10240 个 FP16 元素中 1 个不同,|Δ| = 7.63e-6;闭环后 16/16 greedy-token parity at 1K/4K,acceptance 恢复到 81% 并找回预期 spec 加速)。条件:125B hybrid 模型,SM70/V100 ×4(TP4),FP16,full+piecewise CUDA graphs,`torch.compile`/Inductor。

## 现状与落地
仍 open。作者只请求 scope 决策并表示愿贡献证据、复现方法与测量;处于等待维护者对上述三个问题表态的阶段。

## 关联
- #41758(症状类报告)
- #52522(batch-invariant spec decode,sampling 层 scope)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
