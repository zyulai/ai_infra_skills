# RFC #28663: Deprecate and delete `--moe-runner-backend cutlass`, dense FP4 GEMM, FP4 quantization (in sgl-kernel)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28663](https://github.com/sgl-project/sglang/issues/28663) |
| 作者 | b8zhong |
| 创建日期 | 2026-06-18 |
| 状态 | ⚪ closed(关闭于 2026-07-14) |
| labels | RFC |
| 热度 | 👍 1 · 💬 0 |
| 分类 | MoE |

## 一句话结论
弃用并删除 sgl-kernel 中 `--moe-runner-backend cutlass`、dense FP4 GEMM 与 FP4 量化,收敛到维护更活跃的 Flashinfer 实现。

## 提案动机
该 MoE kernel 最初自 TRTLLM 移植;自从 Flashinfer 集成同类能力后,SGLang 主要调用 Flashinfer 的 API。cutlass 版没有提供任何特殊用法,而 Flashinfer 版本维护更活跃,因此可以删除。(pack 正文仅一小段,以上为正文全部动机内容。)

## 提出的方案
- 弃用并删除 `--moe-runner-backend cutlass`。
- 同时考虑删除 dense FP4 GEMM 与 FP4 量化(同样"没有提供任何特殊用法")。

## 优化点与收益
正文未给出量化数字。收益为维护性:移除重复且缺乏用例的 kernel 路径,把维护收敛到更活跃的 Flashinfer 版本。

## 现状与落地
已关闭(2026-07-14)。正文未附实施 PR,删除是否执行需查关联 PR。

## 关联
正文未列出

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
