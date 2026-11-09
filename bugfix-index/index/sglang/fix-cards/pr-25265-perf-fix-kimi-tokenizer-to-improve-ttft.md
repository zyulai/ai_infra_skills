# PR #25265: [perf] fix kimi tokenizer to improve ttft

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#25265](https://github.com/sgl-project/sglang/pull/25265) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2026-05-15 |
| labels | run-ci |
| 热度 | 👍 5 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | general |

## 一句话结论
tiktoken 派生 tokenizer 被 __call__ 拖入慢速管线(~260ms/请求),改走 .encode() 快路径降 TTFT。

## 问题现象
Kimi-K2.5 的 tokenizer 为 tiktoken 派生,server log 出现 warning(正文附截图)。`TokenizerManager._tokenize_texts` 经 `__call__` 调用 tokenizer,迫使 tiktoken 派生 tokenizer 走 `PreTrainedTokenizer` 的慢速 Python 管线;原生 tiktoken 快路径只有不带 kwargs 的 `.encode()` 才能触达。Microbench(Kimi-K2.5,80k tokens,单条文本):`tok.encode(text)` 47ms vs `tok([text])` 308ms,固定 ~260ms/请求开销,影响所有 tiktoken 派生 tokenizer 的长输入。E2E:`bench_serving --random-input-len 80000 --random-output-len 1 --num-prompts 10 --max-concurrency 1`(Kimi-K2.5-NVFP4,B200×4,TP=4,EAGLE3)Mean TTFT 2948→2791ms、Median 3151→2932ms,约 157ms/请求(~5–7%)。

## 根因
正文披露:`_tokenize_texts` 统一用 `__call__` 编码,tiktoken 派生 tokenizer 因此进不了原生快路径,且 `.encode()` 必须无 kwargs 才走快路径。正文微基准显示分支条件正落在两类 tokenizer 的分界上:Kimi-K2.5 的 `.encode()` 循环 / `tokenizer(...)` 批量比值在 bs=1/2/16/100 均为 6.6×–7.6×(loop 胜),而 Llama-3.1-8B、Qwen2.5-7B 在 bs≥2 时批量更快(0.43×、0.38×,bs=100 约 0.08×);fast tokenizer(Llama/Qwen/BERT)bs≥2 时 `tokenizers`-crate 的 rayon 批量胜出,cross-encoder(BGE-reranker 等)需要 `token_type_ids` 必须走 `tokenizer(...)`。

## 修复方案
`python/sglang/srt/managers/tokenizer_manager.py` 的 `_tokenize_texts`:非 cross-encoder 且 tokenizer 非 fast(`is_fast` 为 False)时,改为逐条 `self.tokenizer.encode(t)`;否则保留原 `tokenizer(tokenizer_input, **tokenizer_kwargs)` 批量路径及 `token_type_ids` 提取:

```diff
+            if not is_cross_encoder and (not getattr(self.tokenizer, "is_fast", False)):
+                input_ids = [self.tokenizer.encode(t) for t in tokenizer_input]
+                token_type_ids = None
+            else:
+                encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
+                input_ids = encoded["input_ids"]
+                token_type_ids = (
+                    encoded.get("token_type_ids") if is_cross_encoder else None
+                )
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:所有非 cross-encoder 的常规编码路径;fast tokenizer(Llama/Qwen/BERT 等)与 cross-encoder(BGE-reranker 等需 token_type_ids)仍走原 `__call__` 批量路径不变;非 fast tokenizer 在 bs≥2 时失去 rayon 批量加速;`.encode()` 分支不接收 tokenizer_kwargs(正文指明快路径需无 kwargs 的 `.encode()`)。

## 关联
- Kimi-K2.5 tokenizer 源码:https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
