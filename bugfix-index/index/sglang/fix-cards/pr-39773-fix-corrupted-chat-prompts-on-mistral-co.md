# PR #39773: Fix corrupted chat prompts on mistral_common tokenizers (tool_choice auto never fires)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39773](https://github.com/sgl-project/sglang/pull/39773) |
| 作者 | alisonshao |
| 合并日期 | 2026-09-18 |
| labels | run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

mistral_common tokenizer 控制 token 无文本形式,render-再编码破坏 prompt;启动探测一次,lossy 时直接编码为 ids 发送。

## 问题现象

Mistral3 模型上 `tool_choice: "auto"` 从不产生 tool call——模型只输出一个 token 即 EOS,返回空内容;而 `tool_choice: "required"` 正常,表面像 tool-calling bug,实际是**每个请求**的 prompt 都被破坏了。对比服务端 prompt token ids 与 `apply_chat_template(tokenize=True)` 的结果:

```
server:   ['<s>', '<', 's', '>[', 'A', 'VAIL', 'ABLE', '_TO', 'OL', 'S', '][', ...]   103 tokens
template: ['<s>', '[AVAILABLE_TOOLS]', '[', ...]                                       82 tokens
```

`[AVAILABLE_TOOLS]`(控制 token id 5)变成字母 `A`,`VAIL`,`ABLE`,`_TO`,`OL`,`S`,且多出一个重复 BOS;模型从未把 tool block 看成控制结构。margin 很薄:首个采样 token `</s>` logprob -1.413,` [` 紧随其后 -1.476,所以表现为"随机"的失败;`required` 通过约束输出掩盖了问题。transformers 也有对应警告:`MistralCommonBackend.apply_chat_template(..., tokenize=False)` 不安全,应使用 `tokenize=True`。普通无工具聊天同样受影响(回复以 `<s>[EVAL]` 开头)。

## 根因

正文披露:`_render_and_encode_chat_template` 把 chat template 渲染成字符串再重新编码。mistral_common tokenizers 会发出没有文本形式的控制 token,这一 round-trip 会破坏它们(变成字面字符并多加一个 BOS)。而每个请求都走文本路径,因为 `_engine_prompt` 在模型 `is_multimodal` 时一律发送 text,而 Mistral3 带 `vision_config`。

## 修复方案

改动集中在 `python/sglang/srt/entrypoints/openai/serving_chat.py`:

1. 启动时调用新增的 `_probe_prompt_text_round_trip()` 探测一次:用 `[{"role": "user", "content": "x"}]` 分别走 render-再-encode 与 `tokenize=True`,比较 ids 是否一致,不一致则判定 lossy(探测抛异常时保守返回 False,保留原有文本路径)。
2. `_engine_prompt`:当探测判定 lossy、且请求不含 image/video/audio(无需占位符展开)且有 `prompt_ids` 时,直接发送 `input_ids` 而非 text;带媒体请求仍走文本路径,因为多模态 processor 需要对文本 tokenize 来展开占位符。
3. `_render_and_encode_chat_template`:lossy 时直接 `apply_chat_template(tokenize=True, ...)` 得到 ids,跳过 render+encode;签名与缓存由 `(rendered_prompt, prompt_ids, decoded_prompt)` 简化为 `(prompt_ids, decoded_prompt)`。

新增单测 `test_chat_prompt_round_trip.py` 覆盖探测与分发逻辑。作者在 Ministral-3-3B(与 Mistral Medium 3.5 同类、同 tekken tokenizer)验证:`tool_choice: "auto"` 从"无 tool call、生成 1 token"变为产出 `get_weather{"location": "Paris, France"}`;prompt 从 103 tokens/双 BOS 变为 82 tokens/控制 token 完整;普通聊天从 21 tokens(回复以 `<s>[EVAL]` 开头)变为 12 tokens 干净回复。探测区分正确:Ministral-3-3B 被判 lossy,Qwen3-0.6B round-trip ids 一致、保留原路径。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:OpenAI serving chat 的 prompt 编码路径。非 lossy tokenizer(如 Qwen3-0.6B)行为不变;lossy tokenizer 上无媒体请求改为发 input_ids,带媒体请求与探测异常场景仍走原文本路径;`_chat_template_cache` 缓存值结构变化,相关旧测试同步更新(`test_serving_chat.py`)。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
