# RFC #33314: [RFC] [Feature]: Apply RFC #8913 to Quantized Linear Methods (Decouple Kernels from Checkpoint Layout)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33314](https://github.com/vllm-project/vllm/issues/33314) |
| 作者 | BadrBasowid |
| 创建日期 | 2026-01-29 |
| 状态 | ⚪ closed(关闭于 2026-09-06) |
| labels | RFC, feature request, stale |
| 热度 | 👍 2 · 💬 10 |
| 分类 | Quantization |

## 一句话结论
把 RFC #8913 的"权重表示与 kernel 解耦"思路落到量化线性层:检查点布局不再决定 kernel,加载归一化为规范表示后由 kernel 侧 repack。

## 提案动机
许多量化 linear method(如 GPTQ/AWQ/FP8/FP4)在 weight-loading 路径(往往是特化的 create_weights)里隐式编码了检查点布局要求,等于检查点格式间接决定了 kernel。后果:create_weights 逻辑在仅 kernel/backend 不同的 method 间重复;kernel 集成难以与其他检查点布局混搭;布局/打包细节泄漏进 create_weights 而非 `process_weights_after_loading` 的职责范围。Kernel 抽象(#27814、#11785)本意已基本与格式无关,但权重加载仍固化了打包假设,限制了可运行的 kernel。

## 提出的方案
- 主流程三步:1) 为量化 linear 层定义规范(canonical)内部权重表示;2) loader/runtime 把各种布局的检查点张量转换为规范表示;3) kernel 集成消费规范权重,可选实现预处理 hook(repack/transpose/reorder)生成 kernel 最优 buffer。
- 接口草图:以 `LinearParams` dataclass(weight / weight_scales / input_scales / zero_points,均为参数类型联合)承载规范表示并内置校验;各 `QuantizationConfig` 提供 `create_linear_params`(如 AWQConfig 产 zero_points,CompressedTensorsConfig 产 input_scales);kernel 以 `can_implement(config, layer_params)` 声明约束(如需 zero_points)。
- 格式不再经 `get_quant_method` 直接映射到 LinearMethod;改为格式特定的权重创建函数,把 format/config 移入 kernel 抽象,使 dispatch 能同时匹配约束与格式。作者倾向彻底移除所有 linear quantization method。
- 实施范围:聚焦预计长期支持、已用 kernel 抽象的 method,包括 Fp8LinearMethod、ModelOptFp8LinearMethod、CompressedTensorsW8A8Fp8/W8A8Int8/W4A8Fp8/W4A8Int/WNA16、QuarkW8A8Fp8、QuarkW8A8Int8、GPTQMarlinLinearMethod(部分 legacy method 已在 #30136 计划废弃)。
- 开放问题:canonicalization + repacking 的内存开销需防 OOM(额外 buffer、dtype 变换、转置);部分混合精度 kernel 打包逻辑紧耦合,需谨慎选规范布局减少转换。

## 优化点与收益
正文未给出量化数字。收益为工程层面:消除重复的 create_weights 代码、kernel 可跨检查点格式复用、新 kernel 或新检查点布局的接入迭代更快;替代方案("不做")被指出会延续 kernel 与格式耦合、重复加载代码与受限复用。

## 现状与落地
已关闭(2026-09-06),labels 含 stale。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- RFC #8913(checkpoint 格式与 kernel 解耦的原始提案)
- #27814、#11785(kernel 抽象)
- #30136(legacy method 废弃计划)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
