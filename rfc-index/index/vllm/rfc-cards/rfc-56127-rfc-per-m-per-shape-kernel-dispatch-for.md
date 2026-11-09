# RFC #56127: RFC: per-M, per-shape kernel dispatch for blockwise-FP8 linear layers (W8A16 at decode, FP8xFP8 at prefill)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56127](https://github.com/vllm-project/vllm/issues/56127) |
| 作者 | Zeuss5 |
| 创建日期 | 2026-09-09 |
| 状态 | ⚪ closed (2026-09-09) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |

## 一句话结论
主张 blockwise-FP8 线性层按 M/shape 分派内核:decode 走 W8A16 流式权重,prefill 走 FP8×FP8,阈值 per shape 而非全局。

## 提案动机
decode batch 下 blockwise-FP8 GEMM 是纯权重流式:无论服务多少 token 都要读 N×K 权重字节,activation 流量(T×K)小两个数量级,MMA 吞吐无关紧要,量化激活零收益还多一次 launch 与约 7× 数值误差;prefill 时 tensor core 成为瓶颈,FP8×FP8 才合适。而 vLLM 目前对 `linear_backend` 只做一次全局选择。

## 提出的方案
- 让每层同时持有两种 prepared weights,按 M 逐调用选择内核,阈值 per shape(`down_proj` 在 T=32 已翻转,`in_proj_z` 同点仍领先 1.05×)。
- 诚实代价:CUTLASS 布局下没有 FP8 权重 × bf16 activation 的免 repack 内核,意味着两格式常驻——Marlin repack 为 int32、与 fp8 原始同大小,该模型 TP=4 约 +6.75 GB/rank,挤占 KV 预算;免 repack 的流式内核才是更有价值的改动。

## 优化点与收益
(条件:4× RTX PRO 6000 Blackwell sm120、PCIe 无 NVLink、Qwen3.6-27B-FP8 TP=4、DRAM-resident 工作集)W8A16÷FP8×FP8 比率:in_proj_z T=1 1.21×、T=4 1.22×;down_proj T=256/1024 为 0.58×/0.54×。精度 2.3e-03…4.5e-03 vs 2.3e-02…3.4e-02(对 fp32 参考,约 7× 更好)。端到端 decode 隔离:batch 1 +5.3%(112.0→117.9)、batch 8 +7.1%(654.0→700.6)、batch 32 +1.6%(noise)、batch 128 −6.4%;独立量化 kernel 234/step → 0。结论限于延迟受限互连,NVLink 上需重测;作者并修正了多处早期错误数字(如 L2 供数导致的 1.81× 实为 1.04–1.22×)。

## 现状与落地
已关闭(2026-09-09,创建同日);正文自述「draft, not filed」且未附 PR,落地情况需查关联 PR。

## 关联
正文未列出 RFC/PR 链接(仅测量方法与仓库内代码描述)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
