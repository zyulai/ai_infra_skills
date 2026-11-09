# RFC #9853: RFC: Command-Line Interface Architecture for SGLang

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9853](https://github.com/sgl-project/sglang/issues/9853) |
| 作者 | vincentzed |
| 创建日期 | 2025-09-01 |
| 状态 | ⚪ closed(关闭于 2026-03-04) |
| labels | RFC, help wanted, high priority |
| 热度 | 👍 10 · 💬 5 |
| 分类 | Architecture / Engine |

## 一句话结论
为 SGLang 引入统一的 `sglang` CLI 替代 python -m 模块调用,并给出可扩展的三层 CLI 架构设计。

## 提案动机
目前用户须直接运行 Python 模块(如 python -m sglang.launch_server、python -m sglang.bench_latency):易打错、难发现,还迫使用户记住日后可能重构的内部模块名;接口把实现细节泄露给用户,使其与包结构耦合。统一 CLI 应让常用工作流显而易见、在内部重构下保持稳定,并把一致的校验与错误信息集中到一处,而非分散在多个 __main__ 块和脚本里。文档深入研究 vLLM 的 CLI,目的是学习其成功经验、避免继承其痛苦。

## 提出的方案
- 命令面:入口 `sglang`,初始命令 launch / chat / complete / batch / bench <subcommand>;benchmark 归为子树(bench latency / serving / throughput)以保持可发现性;launch 还是 serve 待定,选定后另一个做 alias。
- 三层架构:命令框架(推荐 Typer/Click 而非 argparse)、命令注册表(dispatch 支持 lazy import,`sglang --help` 不导入 GPU/模型代码)、配置层(框架无关,把 CLI 选项映射为 typed config 并集中校验)。
- 借鉴 vLLM 三个成功模式:昂贵初始化前 fail-fast 校验、import 纪律(lazy import 保 --help/--version 快)、参数源组合(serving/frontend/engine 选项拼装复用);避免其 argparse 规模化代价:help 过滤层、文档用并行 parser 构造、手动 dispatch 注册。
- 处理器是薄适配器:构造 config → 校验 → 调内部 API;校验必须在昂贵初始化前(含跨 flag 约束、文件存在性、模板兼容);benchmark 用嵌套子命令,`sglang bench --help` 可导入定义而 `sglang --help` 不应导入。
- 向后兼容:模块式调用保留至少一个弃用窗口并告警指向新 CLI,内部改为同一函数的薄 wrapper,不维护两套逻辑。

## 优化点与收益
正文未给出量化数字。性能诉求为设计约束:--version / --help 等琐碎操作保持响应速度,无效配置在模型加载/GPU 初始化前快速失败。

## 现状与落地
已关闭(2026-03-04)。正文状态为 Draft,是否已实现需查关联 PR。

## 关联
正文未列出(正文引用 vLLM 代码路径作分析对象,未附 RFC/PR 链接)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
