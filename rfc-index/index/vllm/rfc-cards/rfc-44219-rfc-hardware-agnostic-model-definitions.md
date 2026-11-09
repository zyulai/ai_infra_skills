# RFC #44219: [RFC]: hardware agnostic model definitions in vLLM

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44219](https://github.com/vllm-project/vllm/issues/44219) |
| 作者 | zou3519 |
| 创建日期 | 2026-06-01 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 40 · 💬 12 |
| 分类 | Hardware / Backend |

## 一句话结论
提议在 vLLM 核心维护社区共治的硬件无关模型定义,让各硬件后端可选复用,兼顾长尾硬件与头部极致性能。

## 提案动机
本 RFC 是 #42770(提议移除整模型 torch.compile、按硬件类型拆分模型定义)的直接后续。#42770 能服务"以 GW 功耗计的头部推理用户"在最热门硬件上跑热门模型的极致性能诉求,但让"支持更多模型 × 更多硬件"变得更难。作者认为这不能覆盖全部用户:社区对替代硬件有浓厚兴趣,给用户更多选择对社区长期有益;而且跨多硬件支持多模型本身很难,在硬件间共享部分技术栈是最简单可靠的方式,同时要设好边界、不拖累头部 GW 用户的优化。

## 提出的方案
- 所有模型在核心仓库维护一份由社区维护的硬件无关模型定义;硬件后端可选择直接使用,也可自建定义(按 #42770 提案);
- 长尾模型:保留现有硬件无关定义与 CustomOp 机制;头部模型:接受硬件无关定义,并通过 PR 逐步 CustomOp 化;
- 目录组织给出两种选项:一是按 `models/deepseek_v4/{nvidia,amd,hw_agnostic}/model.py` 并列存放;二是把现有 `model_executor` 重构为 `model_executor_hw_agnostic`(社区维护),另立 `model_executor_hw_specific`(nv/amd/xpu 追求最佳性能);
- 维护模式:希望由受益方(硬件厂商、模型提供方、PyTorch 团队)协调共同维护;
- 硬性要求:定义须硬件无关(使用原生 torch op + 可按硬件分派的 CustomOp)、支持 torch.compile 全图捕获(TPU、Intel Gaudi、IBM Spyre、CPU 等后端重度受益)、与硬件专用定义相互隔离;保留并允许扩展 CustomOp 机制(可能演进为 vLLM IR 风格管理);
- 备选方案否决理由:HF Transformers 后端存在热门模型开箱不可用(gpt-oss-120b、qwen3.5 等)、无 CustomOp 抽象、不保证全图可捕获、与 stock vLLM 有性能差距;库外维护模型定义则工作量与维护成本明显更高。

## 优化点与收益
收益在于生态与可维护性:共享技术栈降低多硬件 × 多模型的支持成本,同时通过 CustomOp 分发与全图捕获为替代硬件后端(TPU、Gaudi、Spyre、CPU 等)提供接近头部的性能路径,且不阻碍 nvidia 等专用路径的极限优化。正文未给出量化数字。

## 现状与落地
state=open:作者明确反馈周期为 "not sure",处于意见征集阶段;RFC 已 CC 硬件厂商与 PyTorch 相关多位维护者,并致谢协助起草者,尚无落地结论。

## 关联
- https://github.com/vllm-project/vllm/issues/42770 (前置提案:vLLM models proposal)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
