# RFC #14981: [RFC]: vLLM Windows CUDA support

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14981](https://github.com/vllm-project/vllm/issues/14981) |
| 作者 | SystemPanic |
| 创建日期 | 2025-03-17 |
| 状态 | ⚪ closed(关闭于 2025-06-03) |
| labels | RFC |
| 热度 | 👍 18 · 💬 7 |
| 分类 | Feature / Ecosystem |

## 一句话结论
提议让 vLLM 支持 Windows CUDA;正文称支持已在 v0.7.4 完成,等待 PR #14891 审批合并。

## 提案动机
作者列举多类动机:开源 AI 社区——非开发者用户 99% 在 Windows,当前由 llama.cpp/ollama 等后端服务,项目碎片化(服务端类型、内核、量化方案、CUDA 版本互相冲突)造成用户疲劳;被公司要求留在 Windows 的开发者——WSL2 对 PyCharm/VSCode(尤其调试器)不友好,不是现实选项;公共部门常见 Windows Server 与 Azure;学校是 Windows 最多的环境。作者认为 vLLM 是最好的模型服务引擎,应成为各 UI 项目的标准后端,让它们停止重复造轮子。移植与维护成本方面:修改已在 PR 中完成(如 Windows 可用信号、作为库加载的 dll 名称、lambda 内变量改用 `static constexpr` 等),后续贡献者照 PR 即可。

## 提出的方案
- 支持本体:Windows CUDA support 已针对 v0.7.4 完成,等待 PR #14891 审批合并
- CI 构建:最便宜方案是带 Buildkite self-hosted agent 的裸金属 Windows Server + Docker,跑 Windows 容器镜像(choco 装 CUDA + Python,pip 装 torch 及其余依赖);为简单起见建议 GitHub Hosted Runner(不贵、上手成本低),可只在 review 后的 full build check 中执行

## 优化点与收益
正文未给出量化数字。收益为生态覆盖:让 Oobabooga、LM Studio 等 UI 项目不必自行复刻服务与内核维护。

## 现状与落地
已关闭(2025-06-03)。正文写作时点支持代码已完成、待 PR 合并;最终落地情况需查关联 PR #14891。

## 关联
- PR #14891:https://github.com/vllm-project/vllm/pull/14891
- CI 示例(llm.c 仓库 ci.yml):https://raw.githubusercontent.com/rosslwheeler/llm.c/b88f683569423adae51d878f78bfe71df973c6c1/.github/workflows/ci.yml

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
