# RFC #67330: [RFC]: Kernel-owned LoRA fusion for quantized linear layers, starting with gfx1151 W4A16

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#67330](https://github.com/vllm-project/vllm/issues/67330) |
| 作者 | amd-adichatt |
| 创建日期 | 2026-11-12 |
| 状态 | 🟢 open |
| labels | RFC, quantization, rocm |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |

## 一句话结论
提议一个可选的 kernel-owned LoRA fusion 接口:量化线性 kernel 把 LoRA expand 融入自己的 W4A16 GEMM epilogue,消除独立 expand 操作,首发范围是 AMD gfx1151 上 compressed-tensors W4A16 dense linear 层。

## 提案动机
现状下这些层把量化 base GEMM、LoRA shrink、LoRA expand 拆成三个独立逻辑操作;对小的 activation batch,单独的 expand 带来 launch 开销,还要把 base 输出再读一遍、写一遍。原型保留标准 Punica shrink 操作,把 expand 融入 W4A16 GEMM 的 epilogue,从而减少输出内存流量、并在受支持调用上移除独立 expand。目标计算保持不变:对使用 adapter `i` 的 token `t`,仍为 `y_t = x_t @ dequant(W).T + s_i * (x_t @ A_i.T) @ B_i.T + b`;无 adapter 的 token 只取 base 结果;既有 adapter 格式、加载、scaling 与 per-token 选择全部保留。但 fusion 改变了中间舍入与加法顺序,数值等价需要显式验证。作者希望在提交实现 PR 之前,先就共享接口与集成方式征求反馈。

## 提出的方案
1. **小型可选 kernel 接口**:三个 hook——`LinearMethodBase.get_fused_lora_kernel(layer)`、`MPLinearKernel.can_apply_weights_with_lora(layer, x)`、`MPLinearKernel.apply_weights_with_lora(...)`;默认实现保留既有执行路径。量化方法只有当调用该 kernel 能保留其正常 `apply()` 的全部行为时才暴露它;做额外变换或预处理的方法必须保留现有路径,除非显式支持。LoRA layer 在 eligible 时调用可选路径,硬件、shape、rank、layout 限制由 kernel 自己掌握。首发只 opt-in compressed-tensors WNA16 层的 `RDNAHybridW4A16LinearKernel`。
2. **注册 PyTorch custom op**:沿用现有 base GEMM 的注册模式(`direct_register_custom_op`),新增 `torch.ops.vllm.rdna_hybrid_w4a16_lora_apply`:无 active adapter 走 base GEMM;受支持调用走 shrink + GEMM + expand epilogue;被拒绝的调用回退 shrink + base GEMM + expand。op 返回全新输出、声明无输入 mutation、并提供 fake 实现。shape 相关的 fusion 决策留在 opaque op 内部,使编译执行可以处理同时跨 fused 与 fallback shape 的调用,而无需把这些决策 trace 进 Python 控制流。
3. **初始范围收窄**:硬件限 AMD gfx1151;量化前端限 compressed-tensors WNA16 4-bit 权重;FP16/BF16 且 layout 兼容;对称/非对称量化,group size 32/64/128;每 call 激活行数一般 6-64;adapter slot 最多 8;packed projections 最多 3 个 output slice 且边界对齐;TP=1、fully sharded LoRA 禁用。rank 限制取决于 serving buffer 容量与 GEMM tile shape,当前上限 64,更大批次与 tile 时更紧,大 tall projection 另有排除。包络外调用保留标准 LoRA 执行;dual-stream 与 batch-invariant 模式保留现有路径;其他硬件与量化方法维持默认。
4. **初始 opt-in**:新增默认关闭的 `KernelConfig.enable_rdna_hybrid_w4a16_fused_lora = False`,经 `--kernel-config '{"enable_rdna_hybrid_w4a16_fused_lora": true}'` 打开;该字段参与配置 hash,以区分开/关两种编译产物。作者欢迎对配置位置与命名的反馈;并指出该直接 PyTorch operator 独立于 vLLM 的 `CustomOp` class registry,其选择目前不依赖 `compilation_config.custom_ops`。
5. **数值与执行模式验证**:覆盖 stock Punica 对比与独立 dequant 参考推理、可手查的 adapter 校正遗漏/加倍/错路用例、经真实 adapter loader 的 scaling(含 per-projection rank/alpha)、packed projections/部分 adapter/slot 复用/混合 adapter-base batch/inactive-row 污染、fusion eligibility 边界与合法回退配置、custom-op schema/FakeTensor/mutation 与 aliasing/动态 AOT 检查、eager/`torch.compile`/graph replay/并发流,以及真实模型与服务端执行。数值容差需计入 dequant、shrink 累加、cast 以及 bias 与 LoRA 加法顺序的改变,不假设 fused path 的 bitwise identity;实现 PR 前完成数值边界 review,以 fusion flag 作为主要 A/B 控制并固定其他执行设置。

Alternatives considered:保留分离操作(保留既有实现与编译器自由,但 expand 开销仍在)、compiler-pattern fusion(可省去显式 LoRA-layer dispatch hook,但需识别相关 opaque 操作与元数据,eager 执行仍需考虑)、vLLM `CustomOp` wrapper(可提供标准化 backend/配置 dispatch,作者征求其是否在既有量化 kernel 选择之外有附加价值)、backend-specific 特例(可减少通用接口改动,但会把硬件特定行为放进共享 LoRA 代码)。

## 优化点与收益
作者报告的初步结果(单台 native Linux gfx1151 系统,Ryzen AI MAX+ 395 / Radeon 8060S;需随附 reproducibility package 并经最终 review):
- 代表性 fused kernel cases:约 1.06-1.16x stock geometric-mean 速度;M=1-5 与 M>=65 为 1.00x(包络外,走 stock 路径)。
- M=6-32:约 1.08x(FP16)/1.14x(BF16);M=33-64:fused rank-16 约 1.13x(FP16)/1.16-1.17x(BF16),rank 32/64 回落至约 1.00x。
- Serving(Qwen3-4B W4A16 + GSM8K adapter,输入/输出长度随机 128/256,并发 1/8/16/32):并发 1 约 1.00x stock 输出吞吐,并发 8-32 约 1.06-1.07x。
- 部分开启的 FP16 kernel 用例更慢(约 0.90-0.97x stock),因此提议以 opt-in 方式落地;功能关闭时默认行为不变。
- kernel/layer 套件 499 passed、零失败零跳过(如 reported);完整 GSM8K 评估 fusion 84.00% vs stock 84.46%/83.70%,作者明言仅凭这些观察不足以确立 accuracy 等价。
kernel 测例取 Qwen3-4B、Llama-3.1-8B、Qwen2.5-7B 的 qkv/o/gate_up/down 投影,M=1/5/6/8/16/32/33/64/65,rank 16/32/64,round-robin 与混合 adapter 映射,每例取三次重复的中位数。

## 现状与落地
open(2026-11-12 创建,👍 0 · 💬 2);原型已完成,但实现 PR 尚未提交,当前处于征求接口与集成方式反馈的阶段。向维护者提出 6 个问题:可选的 kernel-owned LoRA 接口是否合适或另有更佳集成点、与既有 base GEMM 并列的 registered PyTorch op 是否是首选编译边界、default-off `KernelConfig` 字段是否适合初始部署、应采用什么数值与性能验收标准、在 gfx1151 尚无 upstream CI 覆盖的情况下可接受何种验证安排、应直接走 upstream PR 还是先经 `ROCm/vllm:gfx11` staging。

## 关联
- 无其他 RFC/PR 链接;正文提及的实现落点为 `RDNAHybridW4A16LinearKernel` 与既有 `rdna_hybrid_w4a16_apply` custom op(仓库内代码位置,非 RFC/PR)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
