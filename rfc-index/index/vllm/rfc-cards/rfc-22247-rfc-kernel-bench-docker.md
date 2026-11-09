# RFC #22247: [RFC]: Kernel Bench Docker

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22247](https://github.com/vllm-project/vllm/issues/22247) |
| 作者 | flozi00 |
| 创建日期 | 2025-08-05 |
| 状态 | ⚪ closed(关闭于 2025-12-04) |
| labels | RFC, stale |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |

## 一句话结论
提议提供一个 Docker 快捷命令,以规模化方式生成不同 GPU 与模型组合的 kernel 调优配置。

## 提案动机
正文极简,仅给出两个现有资源链接作为背景:`benchmarks/kernels/benchmark_moe.py`(kernel 基准脚本)与 `vllm/model_executor/layers/fused_moe/configs`(fused MoE 调优配置目录),没有展开具体问题陈述。

## 提出的方案
作者希望能在规模上生成 kernel 配置,因此想要一个 docker 一键命令来完成生成;并询问维护者:若已有类似实现、或对实现风格有偏好请告知,作者愿意据此补充文档,并利用其客户项目中大量不同 GPU 的条件,为不同模型与不同 GPU 版本规模化生成 kernel。Feedback Period、CC List、Any Other Things 三节均为「_No response_」。

## 优化点与收益
正文未给出量化数字,也未给出具体设计;可辨的潜在收益(依正文表述)是把 kernel 配置生成流程容器化,便于跨多种 GPU/模型组合批量产出。

## 现状与落地
已关闭(2025-12-04),labels 含 stale。正文与结尾均看不出是否实现或被替代,落地情况需查关联 PR。

## 关联
正文仅链接 benchmark_moe.py 与 fused_moe/configs 目录,未列出 RFC/PR 链接。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
