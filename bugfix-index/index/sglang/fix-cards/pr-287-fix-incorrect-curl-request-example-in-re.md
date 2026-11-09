# PR #287: Fix Incorrect CURL Request Example in README

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#287](https://github.com/sgl-project/sglang/pull/287) |
| 作者 | amirarsalan90 |
| 合并日期 | 2024-03-13 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

Triton 示例 README 的 CURL 请求体格式有误,改为正确的 `{"INPUT_TEXT": ["harry"]}` 简洁格式。

## 问题现象

`examples/usage/triton/README.md` 中发送给 Triton 服务器的 CURL 请求示例有误。用户照抄该示例向 `http://localhost:8000/v2/models/character_generation/generate` 发请求时会因请求体格式不正确而无法正常工作。作者正文原话:"The CURL request in readme for Triton example had a minor issue, apologies for the mistake."

## 根因

正文未披露根因,只说示例有个小问题。从 diff 看,原示例使用了 Triton KServe v2 协议的完整 `inputs` 数组结构(`"inputs": [{"name": "INPUT_TEXT", "datatype": "STRING", "shape": [1], "data": ["Name1"]}]`),对该服务而言格式不对;修复后改为直接以输入名作键的简单 JSON 体(由 diff 推断)。

## 修复方案

改 `examples/usage/triton/README.md`,替换 CURL 请求体:

```diff
 curl -X POST http://localhost:8000/v2/models/character_generation/generate \
-     -H "Content-Type: application/json" \
-     -d '{
-           "inputs": [
-               {
-                   "name": "INPUT_TEXT",
-                   "datatype": "STRING",
-                   "shape": [1],
-                   "data": ["Name1"]
-               }
-           ]
-         }'
+-H "Content-Type: application/json" \
+-d '{
+  "INPUT_TEXT": ["harry"]
+}'
```

仅改文档示例,不改任何代码。

## 影响与回归风险

- 影响窗口:不影响运行时,仅影响阅读该 README 的用户。
- 回归面:仅 `examples/usage/triton/README.md` 一个文档文件;不涉及代码路径。文末无换行符等格式问题本次也未处理(diff 显示 `\ No newline at end of file`)。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
