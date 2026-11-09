# PR #23401: Fix /generate endpoint crash when sampling params contain null values

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#23401](https://github.com/sgl-project/sglang/pull/23401) |
| 作者 | cctry |
| 合并日期 | 2026-04-22 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | crash |

## 一句话结论
/generate 请求 sampling_params 里带 null(如 top_p:null)会让 verify() 抛 TypeError 崩溃;改为 None 回落到默认值。

## 问题现象
`/generate` 端点在 sampling params 包含 null 值时崩溃。复现命令(正文原文):
```
curl http://localhost:30000/generate -d '{"text":"Hello","sampling_params":{"top_p":null,"max_new_tokens":1}}'
```
崩溃报错为:
```
TypeError: '<' not supported between instances of 'float' and 'NoneType'
```
只要客户端在 sampling_params 中显式传 null(例如 `{"top_p": null}`)即可触发,属于服务端请求处理崩溃。

## 根因
正文披露:此前经 sampling_params 传入的 None 值会直接覆盖 `SamplingParams.__init__` 的构造默认值(即字段被赋成 None 而非默认的 1.0/-1/False 等),随后在 `verify()` 中拿这些 None 值与 float 做比较,抛出 `TypeError: '<' not supported between instances of 'float' and 'NoneType'`。

## 修复方案
按正文所述,在 `SamplingParams.__init__` 内部对 None 做优雅处理:每个非可选(non-optional)参数在收到 None 时回落到其默认值,默认值与既有构造函数签名中的默认值一致。改动文件为 `python/sglang/srt/sampling/sampling_params.py`:
```diff
-        self.temperature = temperature
-        self.top_p = top_p
-        self.top_k = top_k
-        self.min_p = min_p
+        self.temperature = temperature if temperature is not None else 1.0
+        self.top_p = top_p if top_p is not None else 1.0
+        self.top_k = top_k if top_k is not None else -1
+        self.min_p = min_p if min_p is not None else 0.0
```
同样处理的字段还有 `frequency_penalty`、`presence_penalty`、`repetition_penalty`(默认 0.0/0.0/1.0)、`min_new_tokens`(0)、`n`(1)、`ignore_eos`(False)、`skip_special_tokens`(True)、`spaces_between_special_tokens`(True)、`no_stop_trim`(False),并在 `__init__` 顶部加注释说明「对非可选参数,把 None 视为使用默认值,让调用方(如 /generate)可以传 null 而不在 verify() 崩溃」。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有经过 `SamplingParams` 构造的入口(/generate、OpenAI 兼容端点等)共享这次语义变化:显式传 None 的字段现在静默采用默认值,而不是原样保留 None。若有调用方此前依赖「显式传 None 表示某个特殊语义」(而非缺省),行为会改变;其余未列出的可选字段(如 regex、json_schema、custom_params 等)保持原样传递。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
