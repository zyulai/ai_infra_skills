# RFC #17110: [RFC] sgl-kernel Release Strategy Under Current Storage Quota Constraints

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#17110](https://github.com/sgl-project/sglang/issues/17110) |
| 作者 | zhyncs |
| 创建日期 | 2026-01-15 |
| 状态 | ⚪ closed(关闭于 2026-01-20) |
| labels | dependencies, high priority |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Attention / Kernel |

## 一句话结论
PyPI 存储配额仅剩约 10 次发布余量,sgl-kernel 转向以 JIT 更新内核为主、继续复用现有 AOT wheels。

## 提案动机
按当前配额计算:(40−28.4)×1024/(535.6+626.6)≈10。sgl-kernel 项目剩余配额为 40 − 28.4 = 11.6 GB;现有 wheel 体积为 x86 wheel 535.6 MB、aarch64 wheel 626.6 MB——意味着在现行限额下只能再发布少于 ~10 个 sgl-kernel 版本。团队已尝试申请提高总配额,但请求被拒绝。

## 提出的方案
- 优先通过 JIT 更新 kernel。
- 继续复用现有 AOT wheels:它们是大量现有 sglang 版本的依赖,替换会带来显著的下游影响。

## 优化点与收益
正文未给出量化收益数字。核心量化内容即配额账本身:剩余 11.6 GB,每次发布约消耗 535.6 + 626.6 MB(x86 + aarch64 wheel),余量少于 ~10 次发布。

## 现状与落地
已关闭(2026-01-20)。正文未附实施 PR,"JIT 优先 + 复用 AOT wheels"策略的执行情况需查关联 PR 或后续 release 记录。

## 关联
- PyPI sgl-kernel 文件页:https://pypi.org/project/sgl-kernel/#files
- PyPI support 配额工单:https://github.com/pypi/support/issues/8315#issuecomment-3676408738

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
