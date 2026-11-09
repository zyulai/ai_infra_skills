# PR #39160: [Bugfix] Fix extract_hidden_states crash with quantized KV cache dtype

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39160](https://github.com/vllm-project/vllm/pull/39160) |
| 作者 | yubofredwang |
| 合并日期 | 2026-04-07 |
| labels | bug, ready |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
`extract_hidden_states` 把含量化 `cache_dtype` 的全局 cache_config 传给 `CacheOnlyAttentionLayer` 触发 AssertionError;改为传入 cache_dtype 重置为 "auto" 的浅拷贝。

## 问题现象
当 `kv_cache_dtype` 设为量化类型(如 `fp8_e4m3`)时,`ExtractHiddenStatesModel` 把全局 `cache_config` 直接传给 `CacheOnlyAttentionLayer`,而后者只支持 `auto`/`bfloat16`/`float16`,于是以 AssertionError 崩溃。

## 根因
正文披露:`CacheOnlyAttentionLayer` 是为 `extract_hidden_states` speculative decoding 方法存中间 hidden states 的层,并不存真正的 KV 投影,因此量化 cache dtype 对它本就不适用;但它收到的是全局 `cache_config`(带用户的量化 `kv_cache_dtype`),dtype 校验被误触发导致 crash。

## 修复方案
改 `vllm/model_executor/models/extract_hidden_states.py`:在把 `cache_config` 传给 hidden-states cache 层之前,若 `is_quantized_kv_cache(cache_config.cache_dtype)` 为真,用 `dataclasses.replace(cache_config, cache_dtype="auto")` 生成浅拷贝并把 `cache_dtype` 重置为 `"auto"`(解析为模型原生 dtype);目标模型自身的 KV cache 量化配置保持不变。

```diff
+        # Hidden states dtype should be independent of KV cache dtype.
+        if cache_config is not None and is_quantized_kv_cache(cache_config.cache_dtype):
+            cache_config = replace(cache_config, cache_dtype="auto")
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `extract_hidden_states.py` 中 hidden-states cache 层的构造路径。量化 KV cache + extract_hidden_states 组合由 crash 变为可用;非量化 KV cache 时 cache_config 原样传递,行为不变。用 `dataclasses.replace` 生成的是新对象,后续对该 cache_config 的就地修改不会波及全局配置。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
