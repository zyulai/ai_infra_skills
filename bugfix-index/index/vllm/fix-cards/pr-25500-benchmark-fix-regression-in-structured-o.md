# PR #25500: [Benchmark] Fix regression in structured output benchmark

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25500](https://github.com/vllm-project/vllm/pull/25500) |
| 作者 | russellb |
| 合并日期 | 2025-09-24 |
| labels | performance, structured-output, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#22772 挪动了 schema 在请求体中的位置但基准脚本没跟上;改为写入 structured_outputs 嵌套字段。

## 问题现象
结构化输出请求实际上完全不工作("Structured output requests were not actually working"),因为 format spec 没有放到请求体中新的正确位置。正文未附具体报错或数据。

## 根因
正文:这是 #22772 引入的错误("This was a mistake introduced by #22772"),请求体中 format spec 的位置变了,而基准脚本仍按旧位置把 schema 放在 extra_body 顶层键。从 diff 看,修复是把 schema 写入 `extra_body["structured_outputs"][request.structure_type]`,即服务端从 `structured_outputs` 字典读取(由 diff 推断)。

## 修复方案
改 `benchmarks/benchmark_serving_structured_output.py` 的 `prepare_extra_body`:原来 `extra_body[request.structure_type] = request.schema`,改为先建 `extra_body["structured_outputs"] = {}` 再把 schema 挂到其下的 `request.structure_type` 键。

```diff
-        extra_body[request.structure_type] = request.schema
+        extra_body["structured_outputs"] = {}
+        extra_body["structured_outputs"][request.structure_type] = request.schema
```

## 影响与回归风险
- 影响窗口:正文未写(自 #22772 起)。
- 回归面:仅基准脚本的请求构造逻辑(`prepare_extra_body`),不涉及服务端与引擎代码;修复后基准发出的结构化输出请求才会真正生效。

## 关联
- #22772(引入该错误的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
