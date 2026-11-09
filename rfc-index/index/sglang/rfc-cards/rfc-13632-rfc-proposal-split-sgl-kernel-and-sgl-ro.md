# RFC #13632: [RFC] Proposal: Split sgl-kernel and sgl-router into separate repositories

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13632](https://github.com/sgl-project/sglang/issues/13632) |
| 作者 | WingEdge777 |
| 创建日期 | 2025-11-20 |
| 状态 | ⚪ closed (2026-01-20) |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
建议把 sgl-kernel 与 sgl-router 拆到独立仓库(issue 跟踪仍集中主仓库),换取更干净的代码视图与更快、独立的 CI/CD。

## 提案动机
提案源自 discussion #13579。当前混合仓库存在不相关却交叉的提交历史,源码视图不干净;贡献者难以只聚焦自己所属的模块;对 SGL 开发者、维护者与用户的认知负担偏高。

## 提出的方案
- 保持 `/sglang` 集中管理 issue 跟踪
- 将 `sgl-kernel` 与 `sgl-router` 迁移到各自专属仓库
- 预期收益:更干净的源码视图;独立且更快的 CI/CD 管线;贡献者可专注于特定模块;降低各方认知负担

## 优化点与收益
正文未给出量化数字;收益为工程效率与可维护性(独立 CI/CD、聚焦贡献、降低认知负载),无性能类收益主张。

## 现状与落地
已关闭(2026-01-20)。正文为简短建议,未附实现计划;落地情况需查关联 PR。

## 关联
- sgl-project/sglang discussion #13579

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
