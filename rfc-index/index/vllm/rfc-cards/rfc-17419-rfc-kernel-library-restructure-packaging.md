# RFC #17419: [RFC]: Kernel Library Restructure / Packaging Split (addressing long build times)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17419](https://github.com/vllm-project/vllm/issues/17419) |
| 作者 | seemethere |
| 创建日期 | 2025-04-29 |
| 状态 | 🟢 open |
| labels | RFC, keep-open |
| 热度 | 👍 15 · 💬 33 |
| 分类 | Attention / Kernel |

## 一句话结论
把 vLLM 拆成 vllm(引擎)与 vllm-kernels(算子)两个包,解决本地构建过慢并理顺内核与引擎边界。

## 提案动机
vLLM 本地构建耗时过长:CI 上无缓存构建的 wall clock 超过 5 小时。而过去一年内对 vllm 目录的 commit 有 95% 并未改动 csrc 目录,纯属 Python 变更,本不需要额外构建时间。官方安装文档也承认构建时间长,多数人被引导用 python-only 构建来获得可用的工作环境。此外,现有代码结构难以区分内核与引擎,影响了何时该跑性能测试、何时跑单元测试的判断。

## 提出的方案
- 现有包由两条路径构成:csrc(Kernels/PyTorch Custom Operators,含 vllm-flash-attn 源码构建)与 vllm(偶尔使用 csrc 的引擎代码)
- 拆为两个独立 Python 包:vllm(用户安装的引擎包,依赖 vllm-ops)与 vllm-kernels(Kernels + Custom Ops + vllm-flash-attn 等依赖),目录结构一一对应
- 分两步:Step 1 重构为两个包;Step 2 打磨 kernel API,并按后端类型拆分库
- 额外好处:PyPI 包大小分摊到两个包、可增加发布次数;内核与引擎发布节奏解耦(不升引擎也能发内核性能更新);vllm-kernels 可在 cpu/rocm/cuda 等加速器类型间可互换,厂商可自行管理自己的后端包
- 风险与对策:两包版本可能不兼容(如 vllm-kernels==1.0.0 配 vllm==2.0.0),可选硬锁依赖(vllm==2.0.0 依赖 vllm-kernels==2.0.0)或软锁依赖(vllm-kernels>=2.0.0,<=2.1.0)
- 开放问题:仓库目录如何组织、vllm-kernels 是否迁往独立仓库、两包是否始终一起发布

## 优化点与收益
正文给出的数字:CI 无缓存构建 wall clock 时间「exceeds 5 hours」;「95% within the last year」(过去一年 vllm 目录的 commit 未修改 csrc)。其余收益为定性:本地开发者与 CI 的缓存更友好、发布解耦、后端可互换。

## 现状与落地
仍在讨论/待实现(keep-open)。作者无正式反馈期,希望尽快启动(「ideally within the next month or so」),正文处于征集目录结构与兼容性策略意见阶段。

## 关联
- CI 构建时长示例:https://buildkite.com/vllm/ci/builds/18094/steps?jid=01965b50-76d2-4a70-9d55-e517017e0a48
- 从源码安装文档:https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
