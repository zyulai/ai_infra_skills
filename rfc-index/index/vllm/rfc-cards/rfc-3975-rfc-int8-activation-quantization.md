# RFC #3975: [RFC]: Int8 Activation Quantization

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#3975](https://github.com/vllm-project/vllm/issues/3975) |
| 作者 | tlrmchlsmth |
| 创建日期 | 2024-04-10 |
| 状态 | ⚪ closed(关闭于 2024-09-30) |
| labels | RFC |
| 热度 | 👍 10 · 💬 3 |
| 分类 | Quantization |

## 一句话结论
(neuralmagic)提议以增量方式为 vLLM 引入 int8 激活量化,利用 int8 tensor cores 加速 prefill。

## 提案动机
高层目标是通过 int8 tensor cores 提高计算速率来加速 prefill;不预期改善 decode 性能(除非超大批次),因为 decode 时间由加载权重主导,weight-only 量化已能很好覆盖。Int4 激活量化(如 QuaRot)需要更 extensive 的模型定义修改,留作后续;本 RFC 聚焦 Nvidia GPU,其他系统不在范围内。

## 提出的方案
- **增量式路线**:对 PyTorch 模型定义做最小、局部的修改,把 Linear 与 Attention 模块替换为量化对应物而不改动周边计算图;代价是激活需在计算前即时(on the fly)量化。
- **演进方向**:为减少即时量化带来的额外数据搬运,激活需在整个计算图中保持量化态,这要求更广泛、非局部的模型定义修改;将构建量化模型定义的抽象以降低新模型接入成本。
- **算子融合靠 torch.compile**:激活量化引入额外 elementwise 操作,不建议手写融合 kernel,承诺采用 torch.compile 方案(留待未来 RFC 探索)。
- **量化方案取舍**(静态/动态、对称/非对称、per-tensor/per-token、权重量化粒度的性能与精度权衡后,初期支持):
  - 权重:w8a8,静态、对称,per-tensor 或 per-column。
  - 激活:静态、对称或非对称、per-tensor;或动态、对称、per-token。
  - 非对称 w8a8 权重、非对称 per-token 激活等留作未来工作。
- **零点校正项(linear algebra)分析**:对 C=AB 的量化 GEMM 推导 z_A·J_A·B̂、z_B·Â·J_B 等校正项;per-tensor+静态时非对称激活量化可零成本实现(可折入 bias);per-token 情形可用 rank-1 update 避免物化与输出矩阵同尺寸的项。

## 优化点与收益
正文未给出量化数字。预期收益:用 int8 tensor cores 提升 prefill 计算速率;decode 不预期受益(超大批次除外,理由如上)。

## 现状与落地
已关闭(2024-09-30)。正文未说明实现进度,落地情况需查关联 PR。

## 关联
- QuaRot 论文:https://arxiv.org/abs/2404.00456

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
