# RFC #44790: [RFC]: Support Bailing MTP (Multi-Token Prediction) for Ling-2.6-flash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44790](https://github.com/vllm-project/vllm/issues/44790) |
| 作者 | ghost |
| 创建日期 | 2026-06-07 |
| 状态 | ⚪ closed(关闭于 2026-10-07) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
为蚂蚁 Bailing Ling-2.6-flash 的 MTP 机制接入 vLLM 推测解码管线,加速该模型族推理。

## 提案动机
Ant Group 的 Bailing Ling-2.6-flash 模型带有 Multi-Token Prediction(MTP)机制,可在一次前向中预测多个 token 以加速推理。作者指出该架构目前已在 sglang 中获得支持,而 vLLM 尚无对这一 MTP 机制的原生支持;希望补齐这一"适度但有用"的能力,服务依赖该模型族的用户,并征求这一方向是否契合项目发展的反馈。

## 提出的方案
- 模型架构定义:实现 `BailingMTPModel`,处理 Ling-2.6-flash 所需的张量形状与辅助 MTP head。
- 注册与映射:在 `vllm/model_executor/models/registry.py` 注册新架构,使框架能正确把模型分派到 MTP-aware draft worker。
- Draft worker 集成:扩展推测解码 worker,支持 Bailing MTP 特有的状态传递机制,并与 vLLM 现有 PagedAttention、KV-cache 架构保持一致。
- 权重映射:添加转换脚本,把 Bailing 的 MTP 专用权重映射为 vLLM 内部格式,保证"主模型 + MTP head"混合架构正确加载。

## 优化点与收益
正文未给出量化数字(动机仅为"一次前向预测多 token 可加速推理"的定性描述)。

## 现状与落地
已关闭 —— 正文只有提案与"期待反馈"的表述,Feedback Period、CC List、Any Other Things 均为"No response";关闭时正文未写明是否实现或搁置,落地情况需查关联 PR。

## 关联
正文未列出(未附 RFC/PR/文档链接)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
