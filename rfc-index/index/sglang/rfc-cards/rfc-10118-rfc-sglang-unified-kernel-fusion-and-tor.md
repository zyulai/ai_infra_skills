# RFC #10118: [RFC] SGLang unified kernel fusion and torch compile optimisations

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10118](https://github.com/sgl-project/sglang/issues/10118) |
| 作者 | DevashishLal-CB |
| 创建日期 | 2025-09-07 |
| 状态 | ⚪ closed(关闭于 2026-01-04) |
| labels | (none) |
| 热度 | 👍 2 · 💬 13 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
用 torch.compile 自定义 fusion pass 把 kernel 优化与模型架构解耦,模型作者不手写 kernel 调用也能获得优化性能。

## 提案动机
要在多样硬件、workload 与规模上达到最佳性能,常需重度优化的手写 kernel;模型开发者只想专注模型架构,却要用 sgl-kernel 等提供的优化 kernel 重新实现模型——既在多个模型间重复大量 kernel 调用代码,又加重了作者负担。torch.compile 可让编译器代劳:纯 PyTorch 代码加 @torch.compile 即可 trace 出计算图并编译成优化的 triton kernel(取决于 inductor 后端);要榨取最后一点性能时,可用自定义 fusion pass 扩展 torch.compile,pass 与模型架构完全解耦、可跨模型复用。

## 提出的方案
- 用 torch._inductor.pattern_matcher 写匹配/替换规则(正文给出 fused swiglu pass 示例);替换算子注册为支持 torch.compile 的 torch op(direct_register_custom_op + fake impl,fake 实现让编译器验证融合是否适配计算图、避免编译期真执行);可变算子用 auto functionalize v2 完成功能化,省去 v1 的额外拷贝与去功能化 pass。
- 挂载方式:torch._inductor.config.post_grad_custom_post_pass = sglang_fusion_passes,此后任何传给/装饰了 torch.compile 的函数或模块都会执行这些 pass。
- 组件划分:Compilation Config(从 server_args 与环境变量解析配置,config hash 用作缓存失效;含校验与模型特有调整)、Compilation Manager(编译入口,向 inductor post-grad pass 注册 PassManager 并收集统计)、Pass Manager(按配置注册各 pass)、SGLang Inductor Pass 基类(pass 可组合,如同/异量化变体)、算子注册(现有 kernel 与 triton op 注册为 torch op 防止 graph break,如 decode_attention_fwd)。
- 基础能力诉求:可配置(能启用/禁用单个 pass);支持无 CUDA graph 使用(当前 torch.compile 仅支持 cuda graph);inductor 缓存管理加速启动;CUDA graph/piecewise 集成;可观测(pass 运行与替换次数统计);pass 单测易写、可对真实模型测试。
- 候选融合示例:SiLU+Mul(含 +Quant FP8)、UpProject+SiLU+Mul(含 +Quant)、FusedMOE+SiLU+Mul、Attention+Quant(FP8)、RMSNorm+Quant(FP8)、Allreduce+RMSNorm(含 +Quant FP8)等;并概述 vLLM 同类实现(support_torch_compile 装饰器、VllmBackend、piecewise graphs、缓存管理)作参照。

## 优化点与收益
正文未给出量化数字。收益为结构性:kernel 优化与模型架构解耦、fusion pass 跨模型复用,降低模型作者的集成成本。

## 现状与落地
已关闭(2026-01-04)。正文为提案(Implementation Proposed),是否实现需查关联 PR。

## 关联
- vLLM torch compile 博客:https://blog.vllm.ai/2025/08/20/torch-compile.html
- vLLM compilation 实现:https://github.com/vllm-project/vllm/tree/main/vllm/compilation
- SwiGLU 论文:https://arxiv.org/abs/2002.05202
- 自定义算子 torch.compile 支持:https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#adding-torch-compile-support-for-an-operator
- auto functionalize v2 讨论:https://dev-discuss.pytorch.org/t/a-new-strategy-for-automatic-custom-operators-functionalization/2733

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
