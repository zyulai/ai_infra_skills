# RFC #23157: [RFC]: Integrate MetaX GPU Backend into vLLM Plugin

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#23157](https://github.com/vllm-project/vllm/issues/23157) |
| 作者 | ILikeIneine |
| 创建日期 | 2025-08-19 |
| 状态 | ⚪ closed(关闭于 2025-12-18) |
| labels | RFC, stale |
| 热度 | 👍 11 · 💬 2 |
| 分类 | Feature / Ecosystem |

## 一句话结论
请求社区审计 vllm-metax 插件,推动 MetaX GPU(maca 后端)正式并入 vLLM 官方支持。

## 提案动机
MetaX(沐曦)自 vLLM 0.6.6 起持续提供 GPGPU 上的 vLLM 支持,已跑通多种主流模型;早期靠针对每个 vLLM 版本直接改代码、打补丁,随着 vLLM 架构演进复杂化,维护日益困难。在今年 Hardware pluggable RFC(#11162)推出后,MetaX 从 v0.8.5 起把后端迁移为插件方式,集成更模块化但仍需兼容性调整。MetaX 现希望把 maca 后端正式纳入 vLLM 官方支持,获得长期可维护性、顺畅的 CI/CD 流程,并持续跟进上游新特性。

## 提出的方案
- 请求社区审计并评审 vllm-metax 插件是否满足并入 vLLM 官方仓库的条件;插件已适配到 v0.8.5,v0.9.1 当月晚些推送。
- 详细设计文档见 vllm-metax 仓库的 [RFC] vLLM MetaX Plugin Design Doc issue,欢迎对代码提评审意见,团队会持续改进直到满足并入条件。
- 正文坦陈三大限制:1) 包适配——MetaX 只支持偶数版 PyTorch(如 torch==2.4、torch==2.6 及即将的 2.8),奇数版特性可能不兼容(如 torch 2.6 的 `infer_schema` 不支持 `op_func` 参数),设计文档中有临时补丁方案;2) 自定义 op——maca 高度兼容 CUDA,应归为 cuda-alike(类似 ROCm),但所有 kernel 都需为 maca 重编译,且 vLLM 初始化导入 `_custom_ops.py` 时缺少经插件加载特定库的机制,`_C.so`、`_moe_C.so` 等内核库须在插件构建时手动拷贝到 vLLM site-packages;3) `forward_oot`——MetaX 的 Platform 标签设为 `PlatformEnum.CUDA` 而非 OOT(否则需重新继承所有 CustomOp 并覆写 `forward_oot`),在 vLLM 非插件侧加 if-else 分支把 `forward_maca` 重定向到 `forward_cuda` 即可解决。

## 优化点与收益
属于生态/硬件支持扩展而非性能优化:让 MetaX 全栈 GPU(N 系列推理、C 系列通用计算、G 系列图形,基于自研 GPU IP 与独立指令集,软件栈为 MXMACA/maca)用户开箱即用 vLLM,并使插件可长期跟随上游。正文未给出量化数字。

## 现状与落地
已关闭(2025-12-18)。反馈期为 2025 年 8 月起长期;正文只到「请求评审、持续改进插件直至满足并入条件」,是否实际并入官方从正文无法看出,落地情况需查关联 PR/仓库动态。

## 关联
- [vllm-metax 插件仓库](https://github.com/MetaX-MACA/vLLM-metax)
- [MetaX Plugin Design Doc](https://github.com/MetaX-MACA/vLLM-metax/issues/4)
- [RFC: Hardware pluggable #11162](https://github.com/vllm-project/vllm/issues/11162)
- [相关限制讨论 issue 19161](https://github.com/vllm-project/vllm/issues/19161)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
