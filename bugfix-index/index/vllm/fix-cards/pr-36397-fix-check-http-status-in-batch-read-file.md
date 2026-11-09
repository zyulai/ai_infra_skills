# PR #36397: fix: check HTTP status in batch read_file to prevent silent failures

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36397](https://github.com/vllm-project/vllm/pull/36397) |
| 作者 | alvinttang |
| 合并日期 | 2026-03-10 |
| labels | frontend, ready |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
run_batch 的 read_file 不校验 HTTP 状态码,错误页被当 JSONL 输入;补 raise_for_status 修复。

## 问题现象
使用 URL 作为 batch 输入文件(`run_batch.py` 的 `read_file()`)时,若服务端返回非 200 响应(如 404、500),对应的错误页 HTML 会被当作合法的 JSONL batch 输入静默返回,下游随后出现令人困惑的 JSON 解析失败——真正的错误原因(URL 不可用)被掩盖。触发条件:batch 输入 URL 指向不存在或出错的资源。

## 根因
正文披露:`read_file()` 在通过 aiohttp 抓取 batch 输入 URL 时未检查 HTTP 响应状态码;非 200 响应的错误页 HTML 被静默当作有效 JSONL 输入继续处理。同文件中已有的 `download_data()` 函数早已做了响应状态校验,`read_file()` 漏掉了,行为不一致。

## 修复方案
在 `vllm/entrypoints/openai/run_batch.py` 的 `read_file()` 中,`session.get(...)` 之后、读取响应体之前加一行 `resp.raise_for_status()`,使非 2xx 响应直接抛出 `ClientResponseError`,与同文件 `download_data()` 的既有做法保持一致:

```diff
     if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
         async with aiohttp.ClientSession() as session, session.get(path_or_url) as resp:
+            resp.raise_for_status()
             return await resp.text()
```

## 影响与回归风险
- 影响窗口:正文未说明自哪个版本起存在该问题,不予推断。
- 回归面:仅影响 `run_batch.py` 的 URL 输入路径。行为变化是:以前 URL 404/500 时任务进入下游 JSON 解析失败,现在会在读取阶段显式报错;依赖"错误页也能读进来"的用法(几乎不存在)会受影响。本地文件路径分支不变。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
