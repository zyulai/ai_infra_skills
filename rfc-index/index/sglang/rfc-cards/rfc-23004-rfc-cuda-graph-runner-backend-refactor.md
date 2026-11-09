# RFC #23004: [RFC] Cuda Graph Runner Backend Refactor

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23004](https://github.com/sgl-project/sglang/issues/23004) |
| 作者 | Oasis-Git |
| 创建日期 | 2026-04-16 |
| 状态 | ⚪ closed(关闭于 2026-07-07) |
| labels | (none) |
| 热度 | 👍 2 · 💬 1 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
把三套 CUDA Graph 实现重构为 Runner×Backend 两层插拔架构,统一代码结构并支持 (prefill, decode) × (full, breakable, tc-pcg) 任意组合。

## 提案动机
现有三套 cuda graph 实现:full cuda graph、torch-compile 的 piecewise cuda graph(TCPiecewiseCudaGraph)、breakable cuda graph。各 runner 之间重复代码很多;希望重构整体代码结构加以统一,并让用户更容易理解 cuda graph 代码结构。目标是支持灵活的 cuda graph backend 组合,并为 prefill 默认启用 breakable cuda graph。

## 提出的方案
- 两层分离:Runner 层 PrefillCudaGraphRunner / DecodeCudaGraphRunner(继承 BaseCudaGraphRunner,每 phase 恰好一个、实现固定);Backend 层 FullCudaGraphBackend / BreakableCudaGraphBackend / TcPiecewiseCudaGraphBackend(继承 BaseCudaGraphBackend,按 phase 可插拔,由 resolve_{prefill,decode}_backend 在初始化时依 cuda_graph_config[phase].backend 选择)。
- 组合即 cross product:(prefill, decode) × (full, breakable, tcpcg);目标默认模式为 {"decode": "full", "prefill": "breakable"}。
- 参数两条路:JSON config 全控(--cuda-graph-mode {"decode": "full", "prefill": "breakable"})与便捷 flags(--prefill-disable-cuda-graph、--decode-disable-cuda-graph、--prefill-cuda-graph-bs、--decode-cuda-graph-bs 等,翻译为对应 config)。
- API 契约:Runner 向上暴露 can_run / capture / capture_one_shape / replay_prepare / replay;Backend 向下暴露 prepare / capture_session / capture_one / has_shape / runtime_session / replay / cleanup 等钩子(含 prefill 专用的 dummy/serving input 填充)。
- 各 backend 差异:Full 用 torch.cuda.CUDAGraph 捕获整个 forward;Breakable 捕获多段 CUDAGraph 并在 eager_on_graph 断点处缝合;TcPiecewise 用 torch.compile(mode="reduce-overhead") 由其接管执行。
- 扩展点:新增 CUDA Graph 策略只需新增 BaseCudaGraphBackend 子类并在 backend factory 注册,phase runner 代码不变;取代以往"一种策略一个 runner"的设计。

## 优化点与收益
正文未给出量化数字。收益为工程性:消除三套实现间的重复代码,把新策略扩展成本降为"一个子类 + 一处注册"。

## 现状与落地
已关闭(2026-07-07)。正文 Plan 给出三步路径:Breakable cuda graph(#22218)→ cuda graph runner/backend 重构(#23906)→ 默认模式 {"decode": "full", "prefill": "breakable"} 启用(#29458);各步落地情况需查对应 PR。

## 关联
- #22218(Breakable cuda graph)
- #23906(runner/backend 重构)
- #29458(默认模式启用)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
