# RFC #8913: [RFC]: QuantizationConfig and QuantizeMethodBase Refactor for Simplifying Kernel Integrations

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#8913](https://github.com/vllm-project/vllm/issues/8913) |
| 作者 | LucasWilkinson |
| 创建日期 | 2024-09-27 |
| 状态 | 🟢 open |
| labels | RFC, keep-open, unstale |
| 热度 | 👍 8 · 💬 9 |
| 分类 | Quantization |

## 一句话结论
提议把量化 kernel 实现与 checkpoint 格式解耦,简化新量化 kernel 接入 vLLM 的方式。

## 提案动机
当前 vLLM 中 checkpoint 格式与模型执行所用 kernel 普遍紧耦合。随着硬件与 kernel 多样性增长,这种模式带来问题,对量化 kernel(尤其是混合精度 + subbyte 权重)挑战尤甚:为追求性能,量化 kernel 常要运行硬件特化 kernel,混合输入(mixed-input)场景还常把权重预打包成与所运行硬件紧密匹配的定制布局。目标是将 kernel 实现与 checkpoint 格式分离,使一个 kernel 更容易注册为多种 checkpoint 格式的后端,并标准化 vLLM 中量化线性层的调用结构。

## 提出的方案
- 将 `create_weights` 逻辑从 `QuantizeMethodBase` 移入 `QuantizationConfig`,因为后者与序列化格式关联更紧密。
- 新建 `CompressedLinearDescriptor`,让 `QuantizationConfig` 描述所需执行的计算。
- 由 kernel dispatcher 依据该描述选择最合适、能 `can_implement` 该计算的 kernel。
- 整体需要更完善的方式来描述线性层操作,以及更完善的方式描述 vLLM 内的打包布局(packed layouts)。
- 详细设计见正文链接的 Google Docs 设计文档。

## 优化点与收益
收益为工程性/生态性:解耦 checkpoint 格式与 kernel 后,新量化 kernel 更易接入,一个 kernel 后端可服务多种 checkpoint 格式;硬件特化 kernel 与定制权重布局的诉求也能被更系统地表达。正文未给出量化数字。

## 现状与落地
仍开放(labels: RFC, keep-open, unstale)。正文约定反馈期到 Oct 7,作者称将在此前开展准备工作(preparatory work)以作演示;CC 了 dsikka、mgoin、robertgshaw2-neuralmagic 等多位维护者,等待评审与推进。

## 关联
- [设计文档(Google Docs)](https://docs.google.com/document/d/1AfgGfF73H_hcXfw6ehYO_l1vHEItsopbxFoV1PvnGIQ/edit?usp=sharing)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
