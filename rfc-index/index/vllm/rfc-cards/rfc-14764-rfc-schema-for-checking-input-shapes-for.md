# RFC #14764: [RFC]: Schema for checking input shapes for multi-modal models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14764](https://github.com/vllm-project/vllm/issues/14764) |
| 作者 | DarkLight1337 |
| 创建日期 | 2025-03-13 |
| 状态 | ⚪ closed (关闭于 2025-07-25) |
| labels | RFC, feature request, good first issue, multi-modality |
| 热度 | 👍 1 · 💬 32 |
| 分类 | Other |

## 一句话结论
提议新增 TensorSchema 基类,用声明式 shape 标注自动校验多模态模型输入,替代目前近乎缺失的手写检查。

## 提案动机
目前用 `_parse_and_validate_*_input` 校验多模态输入,但只做极少检查,有的模型仅检查输入类型;输入的实际 shape 很容易与 `*ImagePixelInputs` 等类文档中写的 shape 不符,让模型开发者与维护者困惑。

## 提出的方案
- 新增基类 `TensorSchema`,多模态输入类继承它进行校验(示例把 Phi3VImagePixelInputs 从 TypedDict 改为 TensorSchema)
- 用 `typing_extensions.Annotated` 给 tensor 字段附加 `TensorShape` 元数据执行校验,类似 Pydantic 自动校验;为避免性能问题可用 flag 关闭;弃用 Python 3.9 后可切换到 `typing.Annotated`
- 常量维度直接检查(如校验 `data.shape[2] == 3`)
- 同名维度跨字段保持一致(如 data 与 image_sizes 共享的 `bn` 维要求 `data.shape[0] == image_sizes.shape[0]`)
- 字段为 list/tuple 时,用 `len` 检查首维,再递归检查每个元素的其余维度
- 作者建议把它发展为独立 package,惠及 vLLM 之外的项目

## 优化点与收益
正文未给出量化数字。定性收益:输入 shape 校验自动化,减少文档 shape 与实际输入不符带来的开发者/维护者困惑。

## 现状与落地
已关闭。正文未写明最终结果,落地情况需查关联 PR。

## 关联
正文未列出(仅 CC 若干维护者)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
