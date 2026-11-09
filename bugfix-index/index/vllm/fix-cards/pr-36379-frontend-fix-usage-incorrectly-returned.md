# PR #36379: [Frontend] Fix usage incorrectly returned with empty stream_options`

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36379](https://github.com/vllm-project/vllm/pull/36379) |
| 作者 | Csrayz |
| 合并日期 | 2026-03-13 |
| labels | frontend, ready, v1 |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论
streaming 请求传 stream_options={} 时错误返回 usage;把 StreamOptions.include_usage 默认值从 True 改为 False。

## 问题现象
按 OpenAI API 规范(及 #7179),流式模式下只有显式设置 `stream_options={"include_usage": True}` 时才应返回 usage。实际行为:

| 请求 | 期望 | 实际(修复前) |
|---------|----------|---------------------|
| 无 `stream_options` | 无 usage | 无 usage |
| `stream_options={}`(空对象) | 无 usage | 返回 usage |

即空 `stream_options` 对象会错误地携带 usage,影响 OpenAI 兼容 streaming 接口的合规性。

## 根因
正文描述了错误行为与规范要求,未逐字解释代码层原因。(由 diff 推断)`vllm/entrypoints/openai/engine/protocol.py` 中 `StreamOptions.include_usage` 的默认值原本是 `True`,因此空对象 `{}` 反序列化后 `include_usage` 取默认 `True`,被当作用户显式要求返回 usage 处理。

## 修复方案
改 `vllm/entrypoints/openai/engine/protocol.py` 的 `StreamOptions` 模型默认值:

```diff
 class StreamOptions(OpenAIBaseModel):
-    include_usage: bool | None = True
+    include_usage: bool | None = False
     continuous_usage_stats: bool | None = False
```

并在 `tests/v1/entrypoints/openai/test_completion.py` 的 `test_completion_stream_options` 中新增回归用例:`stream=True, stream_options={}` 时断言每个 chunk 的 `usage is None`。正文测试结果为 Passed。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:OpenAI 兼容 serving 的流式 usage 行为——只有显式传 `include_usage: true` 才返回 usage;对本来就显式设置 `{"include_usage": true}` 的客户端无影响。若客户端依赖「传空 stream_options 也返回 usage」的非规范行为,升级后会拿不到 usage。

## 关联
- [#7179](https://github.com/vllm-project/vllm/issues/7179)(正文引用的规范依据)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
