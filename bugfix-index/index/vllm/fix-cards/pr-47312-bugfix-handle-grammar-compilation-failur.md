# PR #47312: [Bugfix] handle grammar compilation failures to avoid engine crash

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47312](https://github.com/vllm-project/vllm/pull/47312) |
| 作者 | izhuhaoran |
| 合并日期 | 2026-07-23 |
| labels | bug, structured-output, ready, v1 |
| 热度 | 👍 3 · 💬 13 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
结构化输出请求的 grammar 编译失败会击穿调度循环、拖垮整个 EngineCore;把错误封装进 Future,由调度器按单请求 FINISHED_ERROR 收尾,其余请求不受影响。

## 问题现象
一条 grammar 无法编译的结构化输出请求会让整个 `EngineCore` 崩溃,而不是只失败该请求:所有在途请求一起被带崩。触发场景:grammar 能通过准入校验但引擎侧编译(xgrammar FSM/token-mask 阶段)失败——具体哪些 schema 会在编译期失败与 xgrammar 版本有关。

## 根因
正文披露:`_create_grammar` 重新抛出异常;因为它在线程池里执行,异常被存进 grammar 的 `Future`,之后调度器在 `schedule()` 内读取 `StructuredOutputRequest.grammar` 时再次抛出——此处没有按请求的守卫,异常逃出忙循环,击落全部在途请求。原 TODO 认为「前置校验已覆盖」的假设是错的,因为两步做的工作不同:准入(`validate_xgrammar_grammar`)只做 schema→grammar(BNF) 转换;引擎侧(`compile_grammar` → `GrammarCompiler.compile_json_schema`)既做转换又编译成 token 级 FSM 与自适应 token-mask 缓存。`from_json_schema` 是 `compile_json_schema` 的严格子集,通过准入不保证引擎侧编译成功。

## 修复方案
- `vllm/v1/structured_output/__init__.py`:`_create_grammar` 捕获后记录日志再抛出;`grammar_init` 同步路径捕获异常装入带异常的 `Future`;移除旧 TODO。
- `vllm/v1/structured_output/request.py`:`_grammar` 类型放宽为可持 `Exception`;`_check_grammar_completion` 把 Future 异常存入 `_grammar`,`grammar` 属性可返回 `Exception`。
- `vllm/v1/core/sched/scheduler.py`:新增 `grammar_compile_error_reqs`;`_try_promote_blocked_waiting_request` 发现 grammar 是 `Exception` 时把请求记入该集合并返回 False;`update_from_output` 将这些请求(与 kv 加载失败请求合并)以 `FINISHED_ERROR` 收尾,并向客户端发 `FinishReason.ERROR` 输出。`finish_requests` 返回值从 `list[tuple[str, int]]` 改为 `list[Request]`,`vllm/v1/engine/core.py` 的 `_send_abort_outputs` 相应适配。
- 新增测试 `test_grammar_compile_error_finishes_only_request`(async_grammar 两种参数),断言坏请求 `FINISHED_ERROR`、健康请求照常调度。

## 影响与回归风险
- 影响窗口:正文未写明版本范围。
- 回归面:影响 V1 结构化输出请求的 grammar 生命周期(同步/异步编译、调度器晋升与收尾路径)与 `finish_requests` 的返回类型签名(调用方 `engine/core.py` 已同步改)。语义变化:编译失败从「进程崩溃」变为「单请求错误」,客户端收到 `finish_reason=ERROR`。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
