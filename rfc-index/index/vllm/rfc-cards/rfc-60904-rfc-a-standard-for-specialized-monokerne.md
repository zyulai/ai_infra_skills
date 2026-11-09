# RFC #60904: [RFC]: A standard for Specialized (MonoKernels) Decodes in vllm/models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#60904](https://github.com/vllm-project/vllm/issues/60904) |
| 作者 | dllehr-amd |
| 创建日期 | 2026-10-09 |
| 状态 | 🟢 open |
| labels | RFC, kimi, quantization, rocm |
| 热度 | 👍 10 · 💬 11 |
| 分类 | Attention / Kernel |

## 一句话结论
为模型专属 MonoKernel decode 建立模块化接入标准,低并发延迟场景获得 2-3x 每 token 时间收益。

## 提案动机
「Specialized Decode」指把模型 forward pass 的大部分/全部操作针对特定硬件上的特定形状与执行模式手工优化实现,通常包含大量 kernel fusion 与重叠以实现低延迟,以及为特定 batch size 调优 grid 的自定义 kernel。这类实现可达数万行代码,复杂度管理是重大挑战。本 RFC 提出在 `vllm/models/<model>/<vendor>/` 下模块化并集成这类运行时的标准方式。其依据是:低并发下 decode 受 launch 开销与权重读取限制,MonoKernel 能同时消除大部分——在 8× MI350X 上,graph-mode GLM-5.2 的一个 decode step 在 vLLM 中发起约 2,459 个 kernel,用 MonoKernel 后仅约 210 个(75 个 MonoKernel launch 加约 135 个用于 dense layers 0-2、输入准备与 head)。

## 提出的方案
七条设计目标:
1. **Opt-in**:默认关闭,一个开关(当前为 `additional_config={"decode_backend": "latency_specialized"}`)。
2. **启动时拒绝**:不支持的部署以一个错误列出全部原因,绝不静默回退。
3. **模型自有代码**:位于 `vllm/models/<model>/<vendor>/specialized_decode/`,不改动核心。
4. **每步一次 `plan()`**:由其选择专用 kernel 或 vLLM 原生 forward;prefill 与混合 batch 始终走原生路径。
5. **每次 pass 一个 `step()`/`run()`**:存在专用 decode 则执行之,否则调用原生实现。
6. **固定 batch sizes**:decode kernel 声明其服务的 batch size,更大的 batch 走原生路径。
7. **避免复制权重**:就地读取 vLLM 已加载的权重与缓存。
- **目录结构**:`model.py`(在 `__init__` 调 `specialized_decode_config()`、forward 中调 `plan()`)、`model_state.py`(Model Runner V2 ModelState,加载后构建 decode 并绑定 KV cache)、`specialized_decode/runtime/`(DecodeConfig、batch sizes、guards 等)与 `mono/`(monokernel 本体);`tests/models/<model>/` 放 refusal 与 plan() 路由的 CPU 测试。
- **两种接入方式**:step 模式(Kimi-K3,#61867:整个 step 在 specialized decode 中运行)与 layer 模式(GLM-5.2,#60889:每层用一个 kernel 或 vLLM 的层)。
- **明确不在范围内**:prefill 与混合 batch;用 `@support_torch_compile` 编译的模型;跨两层的 launch;运行时回退——megakernel 失败引擎应直接停止。
- **启用与拒绝机制**:`--monokernel` 开关(与 `--async-scheduling` 类似,默认关闭);拒绝分三阶段(config 加载前、模型加载后、caches 每次绑定),每阶段报告全部原因;**default-deny**——`supports` 未列出的特性一律拒绝(PP、DP、EP、EPLB、DCP、PCP、LoRA、KV connectors、speculative decoding、sleep mode、CPU weight offload、weight updates 等);MonoKernels 仅在 Model Runner V2 下运行。

## 优化点与收益
(数字照抄正文,条件:GLM-5.2-MXFP4(#60889),TP8,MI350X,~3k-token context;layer times 为 profiler-corrected 单次运行)
- 每 token 时间,batch 1/8,默认 BF16 attention:13.70 / 17.33 ms → 5.92 / 11.44 ms,即 **2.31× / 1.51×**;
- 每 token 时间,batch 1,FP8 attention opt-in:13.70 ms → 4.80 ms,即 **2.85×**;
- MoE 层无 indexer(75 层中 57 层),batch 4/8,FP8:178.8 / 189.6 µs → 69.1 / 100.6 µs,即 **2.59× / 1.88×**;
- MoE 层含 indexer(75 层中 18 层),batch 4/8,FP8:214.9 / 227.6 µs → 88.5 / 123.4 µs,即 **2.43× / 1.84×**;
- AgentX token 间隔,1/8 用户(3,600 s/point),FP8:15.0 / 20.4 ms → 6.5 / 12.7 ms,即 **2.28× / 1.68×**;
- 正确性:GSM8K strict(1,319 题,graphs,BF16)MonoKernel 得 1240,vLLM 三次运行为 1228/1234/1235,「parity: paired McNemar p = 0.22–0.63」。
正文明确:收益随 batch size 增大而缩小,目标是延迟受限的 serving——agents、chat 与小 batch decode。

## 现状与落地
仍在讨论/提案阶段:状态 open,设计为 opt-in 标准,但正文已引用两个在途实现作为参照(Kimi-K3 step 模式 #61867、GLM-5.2 layer 模式 #60889)。

## 关联
- [#60889](https://github.com/vllm-project/vllm/issues/60889)(GLM-5.2-MXFP4,layer 模式与主要性能数据来源)
- [#61867](https://github.com/vllm-project/vllm/issues/61867)(Kimi-K3,step 模式参照)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
