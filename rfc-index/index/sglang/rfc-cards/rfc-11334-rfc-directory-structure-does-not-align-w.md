# RFC #11334: [RFC] Directory Structure Does Not Align with Non-PyTorch Implementations

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11334](https://github.com/sgl-project/sglang/issues/11334) |
| 作者 | chz34 |
| 创建日期 | 2025-10-08 |
| 状态 | ⚪ closed (2025-12-08) |
| labels | (none) |
| 热度 | 👍 10 · 💬 1 |
| 分类 | Other |

## 一句话结论
为 MindSpore/JAX 等非 PyTorch 框架模型实现提出统一的框架专属目录结构与条件依赖安装方案。

## 提案动机
作者团队在为 SGLang 实现 MindSpore backend,同时关注 sglang-jax 进展,预期 SGLang 将集成更多非 PyTorch 框架模型。当前做法(参照 Transformers fallback 思路)是在 `models` 模块加 `mindspore.py` 入口与 `mindspore_models` 目录,首见 PR #9234;这能满足当下需求,但长期看:更多层级实现与引擎 infra 的转换组件不适合永久放在 `models`;其他框架(如 sglang-jax 若与主仓合入)的集成方式也不会局限于 `models` 目录。

## 提出的方案
- 在 `sglang.srt` 下为不同框架建专属目录(`sglang.srt.mindspore`、`sglang.srt.jax` 等),作为 PyTorch 之外的第三方框架实现路径,组件转换或按新框架重实现按需进行
- 核心框架只保留对各 `sglang.srt.PLATFORM` 的最小引用,如 `models/mindspore.py` 仍作为 MindSpore 主入口
- 条件依赖安装:按应用场景提供依赖选项(如 `srt_mindspore`),`pip install -e python[srt_mindspore]`;其余用法(API 调用、配置)与 SGLang 原实现保持一致
- 向社区征询:是否有既有架构调整设计需对齐、目录结构与依赖安装方案如何优化、长期框架扩展性还有哪些遗漏考虑

## 优化点与收益
正文未给出量化数字;收益为架构统一与长期可扩展性:不同框架实现的模型可无缝集成,同时保持与现有 PyTorch 实现的兼容及框架简洁性。

## 现状与落地
已关闭(2025-12-08)。正文止于征求意见的初步提案;落地情况需查关联 PR。

## 关联
- #9234(作者团队的初始 MindSpore 实现 PR)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
