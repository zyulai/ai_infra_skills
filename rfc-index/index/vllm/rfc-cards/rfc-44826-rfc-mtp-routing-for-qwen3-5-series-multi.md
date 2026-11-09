# RFC #44826: [RFC]: MTP Routing for Qwen3.5 Series Multi-LoRA Deployments

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44826](https://github.com/vllm-project/vllm/issues/44826) |
| 作者 | crossxxd |
| 创建日期 | 2026-06-08 |
| 状态 | ⚪ closed(关闭于 2026-10-07) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
为 Qwen3.5 多 LoRA 部署引入 MTP 路由:让不同 LoRA adapter 使用各自专训的 MTP 草稿模型,保住投机解码的加速收益。

## 提案动机
Qwen3.5 系列引入 Multi-Token Prediction(MTP)以提升解码效率、降低推理时延。但在共享基座模型上服务多个 LoRA adapter 的实际部署中,启用 MTP 反而可能带来明显性能退化:部分 LoRA adapter 的 MTP 接受率降低,导致投机解码效率下降、相比非 MTP 路径的吞吐增益缩水。

## 提出的方案
引入路由机制,把请求导向与基座 MTP 模型分开训练的专用 MTP 模型;让多个专门化 MTP 模型共存,分别服务不同的 LoRA adapter 或 adapter 组。关键组件:
- MTP 模型修改:修改现有模型架构 `vllm/model_executor/models/qwen3_5_mtp.py`,使 MTP 可在 graph mode 下动态切换;新增 LoRA MTP 结构,存储 LoRA MTP 的权重及其对应关系。
- 动态路由:依据 LoRA 请求与 LoRA MTP 的对应关系,取出特定 MTP 权重,用于 draft token 的前向计算。

## 优化点与收益
收益为定性描述:在多 LoRA 部署中更好地保留 MTP 加速收益,改善性能与可扩展性;正文未给出量化数字。

## 现状与落地
已关闭(2026-10-07),labels 含 stale;正文未给出后续进展,是否实现需查关联 PR。

## 关联
正文未列出 RFC/PR 链接(仅涉及代码路径 `vllm/model_executor/models/qwen3_5_mtp.py`)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
