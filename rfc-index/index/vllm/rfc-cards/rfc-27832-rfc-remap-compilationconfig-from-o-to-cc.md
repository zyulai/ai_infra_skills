# RFC #27832: [RFC]: Remap `CompilationConfig` from `-O` to `-cc` in CLI

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#27832](https://github.com/vllm-project/vllm/issues/27832) |
| 作者 | ProExpertProg |
| 创建日期 | 2025-10-30 |
| 状态 | ⚪ closed(关闭于 2025-11-28) |
| labels | RFC, good first issue, help wanted, torch.compile |
| 热度 | 👍 10 · 💬 3 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
提议为 `CompilationConfig` 新建 CLI 简写 `-cc`,把 `-O` 语义留给纯优化级别,消除歧义。

## 提案动机
随着 #20283(及 #26847),`-O0`/`-O1`/`-O2`/`-O3` 被重新映射到 `optimization_level` 而非 `CompilationConfig.level`/`mode`,导致 `-O` 的含义取决于其后跟的内容,状态令人困惑,例如:`-O0` -> `optimization_level=0`、`-O 3` -> `optimization_level=3`,但 `-O {"cudagraph_mode": "NONE"}` -> `CompilationConfig(cudagraph_mode="NONE")`、`-O.use_inductor=False` -> `CompilationConfig(use_inductor=False)`。这是糟糕的 UX;不过 CLI 简写仍需保留,方便用户组合不同属性。

## 提出的方案
- 为 `CompilationConfig` 新建简写 `-cc`。备选 `-c` 与 `-C`:如讨论所述,单字母"不 pythonic",大写字母更差(多按 Shift);确切简写开放讨论,用 reaction 投票(👍 -cc / 👎 保持 -O / 🎉 -C / 🚀 -c)。
- 变更示例:
  - `-O {"cudagraph_mode": "NONE"}` -> `-cc {"cudagraph_mode": "NONE"}`
  - `-O.use_inductor=False` -> `-cc.use_inductor=False`
  - `-O0`、`-O 3`、`--compilation-config.backend=eager` 保持不变。

## 优化点与收益
正文未给出量化数字。收益为 CLI/UX 层面:`-O` 只表达优化级别,`CompilationConfig` 有专属简写,减少混淆;改动范围小(标记为 good first issue / help wanted)。

## 现状与落地
已关闭(2025-11-28)。反馈期为一周(10/30 - 11/5)。正文未说明最终实现结果,落地情况需查关联 PR。

## 关联
- https://github.com/vllm-project/vllm/issues/20283
- https://github.com/vllm-project/vllm/pull/26847(含 shorthand 讨论串)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
