# PR #1873: Fix incorrect context length for llama3.2-11b

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#1873](https://github.com/sgl-project/sglang/pull/1873) |
| 作者 | rchen19 |
| 合并日期 | 2024-11-02 |
| labels | (none) |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Llama-3.2 等视觉 LLM 的 context_len 因整份 config 缺 text_config 间接层被误判为默认 2048;改为从 text_config 读取。

## 问题现象
加载 Llama-3.2(llama3.2-11b)这类 vision-LLM 时,模型的 context length 不正确:`TokenizerManager.__init__` 中 `get_context_length()` 找不到最大上下文长度参数,返回默认值 2048,而非模型真实上下文长度。正文未描述更多运行症状与触发日志。

## 根因
正文披露:vision-LLM(如 Llama-3.2)的文本相关配置位于 `config/text_config` 之下,而 `python/sglang/srt/managers/tokenizer_manager.py` 中 `TokenizerManager.__init__` 的这一行

```python
self.context_len = server_args.context_length or get_context_length(self.hf_config)
```

传入的 `self.hf_config` 是整个模型 config 而非 `text_config`,`get_context_length()` 在其上找不到 max context length 参数,于是返回默认值 2048。

## 修复方案
正文提出的思路是:检查整份 config 中是否存在 `text_config` 项,若有则上下文长度参数必然在其下,即在调用侧传入 `getattr(self.hf_config, "text_config", self.hf_config)`。pack 中的 diff 显示实际改动落在 `python/sglang/srt/hf_transformers_utils.py` 的 `get_context_length()`:函数内部自行取 `text_config = getattr(config, "text_config", config)`,rope_scaling 与各上下文长度键均改从 text_config 读取,docstring 亦注明多模态 LLM 应使用 text_config 部分。

```diff
-    rope_scaling = getattr(config, "rope_scaling", None)
+    text_config = getattr(config, "text_config", config)
+    rope_scaling = getattr(text_config, "rope_scaling", None)
...
     for key in CONTEXT_LENGTH_KEYS:
-        val = getattr(config, key, None)
+        val = getattr(text_config, key, None)
```

## 影响与回归风险
- 回归面:`get_context_length()` 是通用的配置读取函数,所有调用方(含纯文本模型)都走新的 text_config 分支;对没有 `text_config` 属性的普通模型,`getattr(config, "text_config", config)` 回退为原 config,行为不变(由 diff 推断)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
