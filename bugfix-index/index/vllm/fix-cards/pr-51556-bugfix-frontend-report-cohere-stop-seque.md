# PR #51556: [Bugfix][Frontend] Report Cohere stop sequences correctly

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51556](https://github.com/vllm-project/vllm/pull/51556) |
| 作者 | taking-lying-flat |
| 合并日期 | 2026-08-11 |
| labels | bug, frontend, ready |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
Cohere v2 端点把匹配停止串的终止误报为 COMPLETE;修复为透传 stop_reason 并映射为 STOP_SEQUENCE。

## 问题现象
正文未给出具体用户报错或日志,只描述影响:Cohere v2 客户端此前无法区分「配置的停止串命中」与「普通完成」——既有的 STOP_SEQUENCE 映射在非流式响应与流式 SSE `message-end` 事件两条路径上都不可达,一律上报 COMPLETE。修复后两条路径都能正确上报 STOP_SEQUENCE(正文 User impact 节)。

## 根因
正文 Root cause 节披露:vLLM 用 `finish_reason="stop"` 统一表示 EOS、stop-token、stop-string 三种终止,匹配到的停止串单独放在 `stop_reason` 字段;而 Cohere v2 的转换逻辑只检查了 `finish_reason`、没有读 `stop_reason`,导致其已有的 STOP_SEQUENCE 映射在两条响应路径上都成为不可达代码。

## 修复方案
改 `vllm/entrypoints/cohere/serving.py`:
- `_map_finish_reason` 增加 `stop_reason: int | str | None` 参数:当 `reason == "stop"` 且 `stop_reason` 是字符串时返回 `"STOP_SEQUENCE"`,以此把匹配的停止串与 EOS、整数 stop-token ID 区分开;
- 非流式路径 `_chat_completion_to_v2` 把 `choice.stop_reason` 传入映射函数;
- 流式状态机新增 `state.stop_reason` 字段,收到 finish_reason 时记录,并在 `_build_message_end_event` 中透传,使 SSE `message-end` 事件同样正确;
- 在 `tests/entrypoints/cohere/` 的三个测试文件中新增非流式与流式回归用例(如 `_map_finish_reason("stop", "<END>") == "STOP_SEQUENCE"`、`_map_finish_reason("stop", 42) == "COMPLETE"`)。

关键 diff:
```diff
+    if reason == "stop" and isinstance(stop_reason, str):
+        return "STOP_SEQUENCE"
     return _FINISH_REASON_MAP.get(reason, "COMPLETE")
```

正文测试结果:pytest 三个 cohere 测试文件 `150 passed`;`ruff check` 与 `ruff format --check` 全部通过。

## 影响与回归风险
- 回归面:按 diff 实际改动,仅涉及 Cohere v2 entrypoint 的 finish_reason 映射函数、非流式转换、流式状态机与 message-end 事件构造。行为变化点是 `finish_reason="stop"` 且 `stop_reason` 为字符串时,上报值由 COMPLETE 改为 STOP_SEQUENCE;正文明确说明本改动只映射已产生的终止元数据,不改变模型执行、token 生成或精度。

## 关联
- 正文提到 PR #45807:处理类似的 Anthropic Messages API 转换问题,与本 PR 的 Cohere v2 端点不同([#45807](https://github.com/vllm-project/vllm/pull/45807))。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
