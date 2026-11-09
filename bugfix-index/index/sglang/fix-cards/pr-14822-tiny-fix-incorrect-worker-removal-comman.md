# PR #14822: Tiny fix incorrect worker removal command

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14822](https://github.com/sgl-project/sglang/pull/14822) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-10 |
| labels | documentation |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
文档中删除 worker 的 curl 示例 URL 未做转义、命令不可用;示例改为 URL 转义形式并补充说明。

## 问题现象
正文的 Motivation 与 Modifications 均为模板占位未填写,正文未描述具体现象。按标题与 diff:文档里「Remove a worker」给出的 curl DELETE 命令中 worker URL 写法有误,照抄会得到错误的移除命令。

## 根因
正文未披露根因。(由 diff 推断)示例里的 worker URL `grpc://0.0.0.0:31000` 含 `:` 与 `/` 等字符,直接拼进 DELETE 路径未经 URL 转义,导致该移除请求写法不正确;正确写法应对 URL 做转义。

## 修复方案
`docs/advanced_features/router.md`:
- 删除 worker 的示例由

```bash
curl -X DELETE http://localhost:30000/workers/grpc://0.0.0.0:31000
```
改为
```bash
curl -X DELETE http://localhost:30000/workers/grpc%3A%2F%2F0.0.0.0%3A31000
```

- 在 legacy endpoints 说明后追加一句:「The worker url in the removal request should be escaped.」

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:纯文档改动(`docs/advanced_features/router.md`),不涉及任何代码路径;仅修正用户照文档执行 `/workers` 删除接口时的写法。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
