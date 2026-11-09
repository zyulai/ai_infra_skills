# PR #33377: [Bugfix][Async][Connector] avoid vllm-side double free during async scheduling + request abort + async KV cache transfer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33377](https://github.com/vllm-project/vllm/pull/33377) |
| 作者 | KuntaiDu |
| 合并日期 | 2026-02-03 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 0 · 💬 9 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
异步调度 + 请求 abort + 异步 KV 传输时同一请求两次进入 _free_blocks 造成 double free;增加 is_finished() 检查跳过已结束请求。

## 问题现象
异步调度(async scheduling)+ KV cache 传输 + 请求 abort 三者叠加时,vLLM 侧同一请求可能两次进入 `get_finished` 并两次进入 `_free_blocks`,造成 double free。测试为 lm-eval 端到端:修复后 lm-eval 在异步调度下可正常完成。

## 根因
正文披露:现有逻辑处理请求 abort 的方式是把 request 置为 `None`,并靠跳过 None 请求来跳过已中止请求。但在异步 KV cache 传输中,请求不会被置 `None`——它仍在 finalize KV cache transfer;对这类请求需要改用 `request.is_finished()` 来判断。

## 修复方案
`vllm/v1/core/sched/scheduler.py` 的 `update_from_output`:对从 KV load 恢复的请求,判断条件由 `request is None` 扩为 `request is None or request.is_finished()`,并加注释说明:当 `delay_free_blocks=True`(KV connector 的异步 KV cache 传输)时,被 abort 的请求不会置 None(以便完成异步传输),此时用 `is_finished()` 检查:

```diff
-            if request is None:
+            if request is None or request.is_finished():
                 # The request is already finished. This can happen if the
                 # request is aborted while the model is executing it (e.g.,
-                # in pipeline parallelism).
+                # in pipeline parallelism or in async scheduling).
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本(标题与正文界定场景为 async scheduling + KV connector + abort 组合)。
- 回归面:`vllm/v1/core/sched/scheduler.py` `update_from_output` 中 KV load 完成请求的收尾路径:已 `is_finished()` 的请求现在与 None 请求一样直接跳过,不再进入后续的块释放逻辑;非异步 KV 传输场景下行为不变(此前这类请求早已被置 None)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
