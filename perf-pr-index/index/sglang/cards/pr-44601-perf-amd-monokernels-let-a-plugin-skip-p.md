# PR #44601: [Perf] [AMD] monokernels: Let a plugin skip per-step attention metadata

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kevin-mii（co-author @kkHuang-amd） |
| 合并 | 2026-10-18 |
| 分类 | Attention / Kernel (tier: major) |
| Label | amd, dependencies, deepseek, run-ci |
| Diff | pack 未提供 diffstat（diff 内容可用：6 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/44601 |

## 一句话结论

给 monokernel 插件声明"前向从不读取的元数据"的能力，DeepSeek-V4.1 spec 借此跳过每步都建又无人读的 low-ratio aiter decode schedule，128k 上下文单请求 decode 快 2.9%。

## 动机

monokernel 前向可能根本不读后端每步构建的 attention metadata。sgl-amd-mono 的 DeepSeek-V4.1 前向用 K2i 给所有 index source 打分，因此其运行的每一步中，HIP radix 后端都会为每个 low-ratio indexer 构建 aiter decode schedule，而没有任何东西读它；构建被录进 decode CUDA graph，每次 replay 都重复执行这个纯开销。

## 优化点

把"插件契约表述为数据"：`MonoSpec` 新增 `unread_metadata` 字段（`frozenset[StepMetadata]`，默认空集，API 版本保持 1，老插件零影响），`StepMetadata` 枚举现含 `LOW_RATIO_DECODE_SCHEDULE`。运行时 `install_monokernel_forward` 把 `(served, spec.unread_metadata)` 挂到 `model._monokernel_unread_metadata`；后端在构建元数据前调 `is_metadata_unread(model, forward_batch, metadata)`——仅当"该步的前向切换确实路由到 monokernel 前向"（`shape_key` 命中 served）且 spec 声明该元数据不被读时才跳过。`deepseek_v4_backend_hip_radix.py` 的 low-ratio decode schedule 构建处（identity-rows 过滤旁边）加上这一判据；不开 `--monokernel-backend` 时一切照旧。附带把 sgl-amd-mono 的版本 pin 收拢进 `monokernels_hip` extra（sgl-amd-mono#5，e07665c），Dockerfile 改为 checkout pip 实际安装的 commit。

## 关键改动

```text
python/sglang/srt/monokernels/spec.py: StepMetadata 枚举 + MonoSpec.unread_metadata
python/sglang/srt/monokernels/runtime.py: is_metadata_unread（shape_key 判路由 + spec 判声明）
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: low-ratio decode schedule 构建跳过 unread 元数据
python/pyproject_other.toml: 新增 monokernels_hip extra 统一插件 pin
```

## 性能声明

MI350X TP4，DSpark（simulated acceptance 3.51），3 轮 x 1024 tokens 中位，每 arm 一个 server，decode 每 output token 毫秒（main ec7ab9d0c2 + plugin 3970534 vs 本 PR 9fd4430dd3 + sgl-amd-mono#5）：8k ctx 请求 1 1.600 -> 1.593 ms（-0.5%）；8k 请求 4 2.396 -> 2.403（+0.3%，噪声）；128k 请求 1 1.643 -> 1.595（-2.9%）；128k 请求 4 2.445 -> 2.407（-1.6%）。两个 skip 运行走同一路径，4 请求时相差至多 2%（作者判为噪声）；单请求时 8k 快 0.4-0.5%、128k 快 1.8-2.9%。未执行 MI355X 与 AgentX serving benchmark。

## 验证与回归风险

CPU 单测 `test_only_routed_steps_skip_the_metadata_a_spec_never_reads` 验证：routed 步才跳过、未构建宽度/无 monokernel/非声明元数据都不跳过；`test/registered/unit/monokernels` 18 passed（后续 commit 29 个 CPU 测试过）。准确度：sgl-amd-mono `test/e2e` 9 tests 于 DSpark verify steps 与 decode server 全过；本 PR 相对 main：DSpark GSM8K 0.905 / KL 0.0139 / long-prompt KL 0.0244（main 0.915/0.0139/0.0273），decode GSM8K 0.905 / KL 0.0115，均在既定限值内。风险：插件与 SGLang 的 StepMetadata 枚举耦合（plugin main 依赖本 PR 先合）；Docker image 本身未构建过（仅 stub 测试了脚本逻辑）；跳过条件依赖 `shape_key` 命中，CUDA graph 下判据在 capture 时固化。

## 关联

上游：kevin-mii/sglang#19（同改动的前身）→ 本 PR；插件侧 sgl-project/sgl-amd-mono#5（DeepSeek-V4.1 spec 声明 schedule unread，先合并）；前序 #44621（Dockerfile 插件安装步骤）；与 #44405、#44630 同分支 main 合并。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
