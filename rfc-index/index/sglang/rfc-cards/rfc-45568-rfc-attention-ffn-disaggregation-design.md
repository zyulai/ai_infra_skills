# RFC #45568: [RFC] Attention-FFN disaggregation: design and integration plan

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#45568](https://github.com/sgl-project/sglang/issues/45568) |
| 作者 | sunfanyi |
| 创建日期 | 2026-10-21 |
| 状态 | 🟢 open |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Disaggregated Serving |

## 一句话结论
提出以 8 个小 PR 的序列把 Attention-FFN 分离 (AFD) 集成进 SGLang:A/F 分组独立扩容、共享 expert 权重;完整实现已在 #45562。

## 提案动机
AFD 把 attention 与 routed experts 放到不同 GPU 组,多个 attention 实例可共享一个 FFN 组及其 routed-expert 权重,使 attention 与 expert 容量独立供给;跨 attention 实例池化 token 可改善 expert batching,microbatch overlap 可隐藏部分通信与远端执行延迟;收益取决于 workload、资源分配与通信开销。

## 提出的方案
- 角色边界:A(attention workers)拥有请求调度、KV cache、attention、routing、shared experts、residual 与 sampling,投机 draft 留在 A;F(FFN workers)只执行 routed experts 并返回输出,不调度生成请求、不维护 KV cache
- 启动握手校验模型、expert layout、量化与拓扑兼容性,并为非对称 A/F 组协商通信与 graph 容量;成员启动时固定;A/F 各自 capture 本地 CUDA graph,不跨进程共享 graph 对象
- 传输演进:Staged NCCL 起步(标准依赖)→ NVSHMEM(GPU 侧通信与同步)→ Direct DeepEP 与 MegaMoE M2N 从 A 直连路由 token 到 expert 所在 F rank 以减少 staged 传输;可选组件仅按所选 backend 加载
- 通过 `--afd-role` opt-in;实现集中于 `srt/afd`,集成点在角色加载、调度、graph 执行与模型 forward
- 8 步 PR 序列:#45958(core AFD 与非对称并行)→ CUDA Graph → decode microbatch overlap → 多 A 共享 F → NVSHMEM → Direct DeepEP → Direct MegaMoE → EAGLE/MTP + DSpark
- 已覆盖:Qwen3 MoE、DeepSeek V3/V3.2/V4、GLM-5 MoE DSA,含 EAGLE/MTP、DSpark 及与 PD 分离的组合(A 的 decode 角色);初始约束:固定成员、同质 attention 实例、A 内 attention TP1、Direct backend F 侧纯 EP;后续:Qwen3.5 MoE、F 侧 PP、EPLB、弹性 A/F 伸缩

## 优化点与收益
正文未给出量化数字;精度与性能结果在 #45562 的 PR 描述中,并绑定其测量 revision(SGLang `34a32688a4d29082f936a7a068e579aa7465b82c`、DeepGEMM `d52c4444086268983f9ee55ba6543b26dd509838`),不代表中间 PR 或更新基线的结果。

## 现状与落地
仍开放:完整实现已在 PR #45562(附精度结果、性能测量与复现命令),首个实现 PR #45958 已提交;征求对运行时接口、PR 边界与验证要求的反馈。

## 关联
- #45562(完整实现 PR)
- #45958(首个实现 PR)
- DeepEP #14(disaggregated expert dispatch/combine)
- DeepGEMM #110(MegaMoE M2N)
- #9401(早期 AFD 讨论)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
