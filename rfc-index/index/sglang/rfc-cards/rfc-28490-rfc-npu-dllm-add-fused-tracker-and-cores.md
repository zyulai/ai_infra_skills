# RFC #28490: [RFC][NPU][DLLM] Add fused tracker and coreset kernels for LLaDA2 JointThreshold

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28490](https://github.com/sgl-project/sglang/issues/28490) |
| 作者 | leedongkun30-arch |
| 创建日期 | 2026-06-17 |
| 状态 | ⚪ closed (2026-08-19) |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |

## 一句话结论
为 LLaDA2.1-mini dLLM 的 JointThreshold 解码路径新增两个 NPU 融合 kernel,消除 per-layer 小算子发射开销。

## 提案动机
JointThreshold 解码循环中,每个 MoE 层每次去噪前向都要执行:①history-gate tracker max + EMA 更新;②每次调用的 coreset keep-mask 选择(softmax-sum 打分 → 组平衡 top-q)。在 NPU 上这两者是 dispatch-bound:各自由一小组小 kernel 构成,host 发射成本主导其计算,设备在发射间隙空转。把每条路径融合成单 kernel,即可在不改变选择语义的前提下把这些发射从解码关键路径上移除。

## 提出的方案
- moe_gate_max_update:把每专家 running-max 更新与 EMA 更新融合为一个 NPU kernel(替换每层 torch.maximum + mul_/add_ 一对操作)。
- moe_shortlist_mask:一次 HBM 往返内完成 coreset 选择(bf16 舍入的 softmax-sum 打分;组内精确 top-q、最低下角标 tie-break,匹配 torch.topk(q)),写 keep-set 并掩掉被弃专家(替换每层约 6–10 个小向量算子)。
- 两 kernel 均 NPU-only,针对当前 LLaDA2 MoE(E=256,8 groups)shape 特化;Python 集成仅在启用时使用,否则走原有 Torch 路径。
- 上游顺序:kernel 源 PR 进 sgl-kernel-npu(含绑定与正确性测试)→ maintainer 发版 → sglang 侧 Python 集成 PR(env 门控,依赖含新 op 的 release)。
- 守护式回退:检查 device.type == "npu"、fp32 dtype 与 contiguity、op 可用性(hasattr(torch.ops.npu, ...))与运行时 flag,任一不满足即走 Torch 路径;模型在 kernel 缺失时仍正常加载服务,CUDA/默认执行结构上不受影响。
- 明确排除:cbout、maxfold、tracker sum-update、deferred-flush 变体——本地测过但不是可复现的发布候选。

## 优化点与收益
GSM8K (n=50),Ascend 910B3,LLaDA2.1-mini,SGLang dLLM JointThreshold;对照组仅关掉两个新 kernel flag(TRACKER_MAX_UPD_KERNEL=0、PERCALL_KEEP_KERNEL=0),其余配方相同:
- Control(kernels off):accuracy 46/50 = 92.0%,output TPS 355.107,elapsed 42.68 s。
- 开 kernel:accuracy 46/50 = 92.0%,output TPS 373.346,elapsed 40.60 s。
- Δ = +18.24 tok/s (+5.14%),准确率完全相同。作者计划扩展到 n ≥ 100 并补 per-op 回退等价测试。
- 环境:CANN 9.0.0,torch/torch_npu 2.10.0,bfloat16 / ascend / max-running-requests 1。

## 现状与落地
已关闭(2026-08-19)。正文给出 kernel → release → integration 的上游序列与两个开放问题(kernel 放现有 sgl-kernel-npu 布局还是 dLLM 专用子模块;发版/版本钉定机制),未写明各 PR 是否已合并,落地情况需查关联 PR。

## 关联
- #27127(guarded NPU-only fallback): https://github.com/sgl-project/sglang/issues/27127
- #21094(before/after dLLM 吞吐证据): https://github.com/sgl-project/sglang/issues/21094
- #18485(how-to-run + benchmark 证据): https://github.com/sgl-project/sglang/issues/18485
- #27144(NPU dLLM PR + GSM8K 证据): https://github.com/sgl-project/sglang/issues/27144

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
