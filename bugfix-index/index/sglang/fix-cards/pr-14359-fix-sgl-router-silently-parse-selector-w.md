# PR #14359: Fix sgl-router silently parse selector wrongly causing OME fail to discover pods

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14359](https://github.com/sgl-project/sglang/pull/14359) |
| 作者 | fzyzcjy |
| 合并日期 | 2025-12-04 |
| labels | run-ci, model-gateway |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
sgl-router 对 `--selector a=b c=d` 这种单字符串多 k=v 写法静默解析出错,OME 场景发现不了 pods;遇到含空格的单项先按空格拆分。

## 问题现象
正文 Motivation 仅一句话:"this will not discover pods",并给出 OME 的部署参数样式:
```
        - --selector
        - component=engine ome.io/inferenceservice=$(INFERENCESERVICE_NAME)
```
即两个键值对合在同一个 YAML 列表项里、以空格分隔传入 `--selector`。此场景下 sgl-router 的服务发现发现不了 pods,且(按标题)不报任何错误——是「silently parse selector wrongly」。正文未描述其他触发条件或影响面,可推断所有用这种单字符串写法传 selector 的用户都会受影响。

## 根因
正文未披露具体解析逻辑,只在标题与示例中指出 selector 被静默错误解析。由 diff 推断:`_parse_selector` 逐项处理 `selector_list`,只有含 `=` 的项才进 selector;整个 `component=engine ome.io/inferenceservice=NAME` 字符串被当作一个 item,未被拆成两个键值对,得到的 selector 与用户意图不符且不抛错, pods 因此匹配不到。

## 修复方案
`sgl-router/bindings/python/sglang_router/router_args.py` 的 `_parse_selector` 增加对单项含空格的处理:
```diff
+        # Support `- --selector\n- a=b c=d` case
+        if len(selector_list) == 1 and (" " in selector_list[0]):
+            selector_list = selector_list[0].split(" ")
```
即只有一个元素且含空格时,先按空格拆分为多个 k=v 项再解析。
`sgl-router/py_test/unit/test_arg_parser.py` 的 `test_parse_service_discovery_args` 新增 `args_b` 用例(`"app=worker env=prod"`,注释注明 "OME has this style"),断言解析结果为 `{"app": "worker", "env": "prod"}`,原多元素写法用例保留为 `args_a`,两种样式循环断言。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅影响 sgl-router Python 侧参数解析中 `--selector` 的处理;新增分支只在「列表长度为 1 且含空格」时触发,原有的多元素写法与不含空格的单项解析路径不变(diff 可见),不会改变已正常工作用户的 selector 语义。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
