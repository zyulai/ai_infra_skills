# PR #20983: fix(frontend): always include usage, when configured to do so

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#20983](https://github.com/vllm-project/vllm/pull/20983) |
| 作者 | max-wittig |
| 合并日期 | 2025-10-14 |
| labels | frontend, ready |
| 热度 | 👍 2 · 💬 35 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
`--enable-force-include-usage` 在未指定 `stream_options` 的流式请求上不生效(usage 仍省略);抽出共享 `should_include_usage`,无 stream_options 时也按开关强制返回 usage。

## 问题现象
配置 `--enable-force-include-usage` 后,未指定 `stream_options` 的流式请求末尾仍不包含 usage 字段。测试计划:以该开关启动 vLLM,发送普通流式请求,观察末尾 usage 是否被包含。正文未描述具体报错。

## 根因
正文披露:这是 #19695 的跟进修复——该 PR 实现 force-include-usage 时没有考虑「未指定任何 stream_options」的情形,其分支在 `stream_options` 为空时直接把 `include_usage` 置 False,开关随之失效。

## 修复方案
- 在 `vllm/entrypoints/utils.py` 新增共享辅助函数,统一 chat/completion 两条流式路径的判定:

```python
def should_include_usage(stream_options, enable_force_include_usage):
    if stream_options:
        include_usage = stream_options.include_usage or enable_force_include_usage
        include_continuous_usage = include_usage and bool(
            stream_options.continuous_usage_stats
        )
    else:
        include_usage, include_continuous_usage = enable_force_include_usage, False
    return include_usage, include_continuous_usage
```

- `serving_chat.py` / `serving_completion.py` 的流生成器删除各自内联的 if/else,改调该辅助函数;`enable_force_include_usage` 参数从基类 `OpenAIServing.__init__` 与各流生成器签名中移除,改由 `serving_completion`、`speech_to_text` 等子类自行持有。
- `speech_to_text.py` 的 `include_usage` 改为 `self.enable_force_include_usage or request.stream_include_usage`。
- `api_server.py` 与 `run_batch.py` 在构造 chat/transcription 等 serving 对象时传入 `args.enable_force_include_usage`;`run_batch.py` 新增对应 CLI 参数。
- 新增 `tests/entrypoints/openai/test_enable_force_include_usage.py`(chat 与 whisper 转写两条流式用例),并在 `pyproject.toml` 注册 `extra_server_args` 测试 marker。

## 影响与回归风险
- 影响窗口:正文未写明版本范围;该行为由 #19695 引入,本 PR 为其跟进修复。
- 回归面:OpenAI 兼容前端的流式响应路径(chat、completion、transcription/translate)及 batch 运行器;未开启开关的服务器行为不变,开启后未带 `stream_options` 的流式请求会在末尾多出 usage chunk。构造函数签名变化影响 serving 子类实例化路径。新增测试需启动 Qwen/Qwen3-0.6B 与 openai/whisper-large-v3-turbo 服务。

## 关联
- #19695(前作,本 PR 为其跟进修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
