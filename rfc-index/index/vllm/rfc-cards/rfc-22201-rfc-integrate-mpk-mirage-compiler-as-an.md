# RFC #22201: [RFC]: Integrate MPK (Mirage) compiler as an experimental execution backend to vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22201](https://github.com/vllm-project/vllm/issues/22201) |
| 作者 | ProExpertProg |
| 创建日期 | 2025-08-04 |
| 状态 | ⚪ closed(关闭于 2026-07-02) |
| labels | RFC, stale, torch.compile |
| 热度 | 👍 12 · 💬 10 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
把 megakernel 编译器 Mirage 作为实验性 torch.compile 后端接入 vLLM,降低前向延迟。

## 提案动机
LLM 推理(尤其是中小模型)仍未达到理论 TFLOPs 与内存带宽上限,原因包括 CPU 开销、kernel launch 开销以及 GPU kernel 执行尾部延迟等流水线气泡。CUDA Graphs 有一定帮助,但一个有前景的方向是编译使用 precision scheduling 的 megakernel:megakernel 是单个 GPU kernel,SM 读取并执行对应不同计算/内存操作的 "tasks",凭借更好的操作重叠、更少的内存加载和更高的硬件利用率,可消除上述延迟。MPK/Mirage 就是这样一种 megakernel 编译器,输入其自定义 DSL 的模型定义,产出覆盖整个模型 forward pass 的单个 megakernel。

## 提出的方案
- **总体架构**:新增一个 Dynamo-based 后端,把 vLLM 产出的 `fx.Graph` 编译为 MPK megakernel;由两层组成——兼容层(位于 vLLM 内,把 vLLM 产出的图转换为 Mirage 友好的图)与 Mirage 后端(位于 `mirage` pip 包,把第二张图编译为 megakernel);未来任一侧变化都只需在兼容层吸收,无需改动另一侧。
- **算子映射**:Mirage 偏好接收高层算子(RMSNorm、RoPE、quant 等)而非低层 torch 操作,兼容层负责把 vLLM custom ops 转为 Mirage 高层 op(如 `torch.ops._C.rms_norm` -> `torch.ops.mirage.rms_norm`)。
- **Paged Attention**:为 Mirage 新增一个 attention backend——它不执行 attention(或为正确性回退到 flex attention/手写 torch 实现),而是在 Dynamo trace 时产出 Mirage 友好的 attention 表示;attention metadata builder 的用法与 full cudagraphs 相同,static buffer 在 forward 之前填充,使 Mirage 无需关心传递 attention metadata 的 CPU 逻辑;buffer 传入方式首选「完全 trace 进 `fx.Graph`」,不行再回退「编译时手动传入」。
- **编译集成**:无需改动编译逻辑——用 Dynamo(非 vLLM compile)时本就接受自定义 Dynamo backend,在 `CompilationConfig.backend` 传 `"mirage"` 即可;`DYNAMO_ONCE` 编译层(#20283 之后)的 shape 信息传播问题可能需处理;跳过 `VLLM_COMPILE` 则没有缓存,留待后续 RFC(需 Mirage 支持缓存)。
- **降级与扩展**:无法编译的区域可直接执行其 `fx.Graph` 表示,常见的话也可手动交给 Inductor;支持只编译特定子图;cudagraph capture 非必需(只启动单个 kernel),但借助 #20059 在图被不支持的算子分裂成多个 megakernel 时仍可用 full cudagraphs;这套基础设施未来可复用于其他采用 precision scheduling/megakernel 的实验编译器。
- **明确不考虑**的替代方案:作为 Inductor backend 接入、用 pattern matcher 做模式替换、维护独立的 Mirage 模型定义、以及在 GPU 上做 batching/scheduling 决策(正文认为过于实验性)。

## 优化点与收益
照抄正文引用:「Mirage demonstrates a 20-25% improvement in forward pass latency over vLLM」——来源为 Mirage 官方博客(条件:博客数据而非 RFC 内的测试,作者同时在帖中请 Mirage 团队补充更全面的 benchmark)。

## 现状与落地
已关闭(2026-07-02),labels 含 stale;正文仅提出设计与 Feedback Period(8/4 - 8/15),未写明是否已实现或被替代,落地情况需查关联 PR。

## 关联
- [Mirage 博客:Compiling LLMs into a Megakernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)(20-25% 数据出处)
- PR [#20283](https://github.com/vllm-project/vllm/pull/20283)(`DYNAMO_ONCE` 编译层相关)
- [#20059](https://github.com/vllm-project/vllm/issues/20059)(full cudagraphs 支持)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
