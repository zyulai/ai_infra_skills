# PR #67153: [Perf][Frontend] Seed the fast detokenizer with a prompt tail, not the full prompt

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mawong-amd |
| 合并 | 2026-11-19 |
| 分类 | Sampler / Detokenization(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67153 |

## 一句话结论

`FastIncrementalDetokenizer` 此前用整个 prompt 预置 `DecodeStream`,每个请求首个输出 token 都要在输出处理循环上解码整个 prompt(100k tokens 时 17 ms,byte-fallback tokenizer 约 40 ms),阻塞所有其他流;本 PR 改用能精确复现首 delta 的短 prompt tail,首个输出 token 从 17.2-17.5 ms 降到 0.005 ms。

## 动机

`DecodeStream` 在第一次 `step()` 时解码全部 seed ids,而 seed 只影响第一个 delta(其后只保留新 id)。全 prompt 预置使长 prompt 的第一个输出 token 付出 O(prompt) 时间,并阻塞同一输出循环上的每个其他 stream。难点在于:Metaspace、WordPiece、leading-`Strip` 等 decoder 对文本开头做特殊处理(如丢前导空格),被 skip 的 special token 等不解出文本的 token 不能计入 seed;tail 文本若以 U+FFFD 结尾(tail 从 prompt 最后一个字符中间开始)需延长;prompt 自身以 U+FFFD 结尾时(预存在问题 #48853)只有全 prompt seed 能保持原输出。

## 优化点

- 新函数 `get_decode_stream_seed(tokenizer, prompt_token_ids, skip_special_tokens)`:从 prompt 尾部往回取 5 个(`INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET`)能单独解出文本的 token;tail 文本以 U+FFFD 结尾则最多再延 3 token(`_MAX_UTF8_SEED_EXTENSION = 3`,UTF-8 字符至多 4 字节);walk 上限 256 token(`_MAX_SEED_WINDOW = 256`),超限或 prompt 以 U+FFFD 结尾时回退全 prompt。无新 config/env。
- `FastIncrementalDetokenizer.__init__` 的 `DecodeStream(ids=...)` 改为传入该 seed。
- 取代 #51281 与 #57330(停滞)的固定 32/128-token tail:按 @njhill 的 review,固定 tail 不总安全——以 32+ 个被 skip 的 special token 结尾的 prompt 在 Metaspace/WordPiece/`Strip` 下会丢首个输出 token 的前导空格;本 PR 只数解出文本的 token 来避免。

## 关键改动

```python
# vllm/v1/engine/detokenizer.py
_MAX_UTF8_SEED_EXTENSION = 3
_MAX_SEED_WINDOW = 256

def get_decode_stream_seed(
    tokenizer: Tokenizer,
    prompt_token_ids: list[int] | None,
    skip_special_tokens: bool,
) -> list[int] | None:
    """Return the prompt suffix used to prime a `DecodeStream`.

    `DecodeStream` decodes all of its seed ids on the first `step()`, so
    seeding it with the whole prompt costs O(prompt) time. ...
    """
    if not prompt_token_ids:
        return prompt_token_ids
    start = len(prompt_token_ids)
    min_start = max(0, start - _MAX_SEED_WINDOW)
    needed = INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET
    for _ in range(_MAX_UTF8_SEED_EXTENSION + 1):
        while start > min_start and needed > 0:
            start -= 1
            if tokenizer.decode(
                [prompt_token_ids[start]], skip_special_tokens=skip_special_tokens
            ):
                needed -= 1
        if needed > 0:
            break
        window = prompt_token_ids[start:]
        text = tokenizer.decode(window, skip_special_tokens=skip_special_tokens)
        if not text.endswith("\ufffd"):
            return window
        needed = 1
    return prompt_token_ids
```

## 性能声明

- 首个输出 token(FastIncrementalDetokenizer 构造 + 首次 decode_next,100k-token prompt,21 次中位数,EPYC 9575F 单核):`Qwen/Qwen3-0.6B`(byte-level BPE)17.2-17.5 ms → 0.005 ms;`codellama/CodeLlama-7b-hf`(byte fallback)39.4-41.6 ms → 0.007 ms;`amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` 18.6-18.7 ms → 0.005 ms。
- 服务(Muse Spark Open,TEP4、FP8 KV、DSpark k=7,4x MI355X,`--max-num-batched-tokens 16384`,两臂同时各占半节点再互换):100k uncached 4 前缀 output tok/s 38.0/37.9 → 39.2/39.4,ITL p99 1129/1093 → 988/993 ms(约 -11%);32k uncached 123.2/124.6 → 125.7/126.2,p99 812/803 → 786/790;100k cached 170.4/147.8 → 167.4/185.1,p99 148/138 → 63/62 ms。
- ITL p50 在两个 100k 行高 2.5-4 ms、32k 在噪声内——作者说明是记账效应不是解码变慢(单流 150k prompt:TTFT 153 → 90 ms,末 token 1159 → 1157 ms 不变);同臂两对吞吐差可达 15%(cached 行,main),吞吐差异在 run-to-run 噪声内,输出吞吐无回归。

## 验证与回归风险

- `pytest tests/detokenizer/test_decode_stream_seed.py`:121 passed(CPU,下载 11 个 ungated tokenizers)。新测试对每个 prompt cut(含多字节字符、组合标记、emoji、whitespace run 的中间)+ 0-40 个 special token、两种 `skip_special_tokens`、随机 token id,要求 tail seed 与全 prompt seed 流出完全相同的 delta;非 U+FFFD 结尾的 seed 至多 8 个含文本 token。若 seed 函数换成 `prompt_token_ids[-5:]` 则 35/121 失败。
- `pytest tests/tokenizers_/test_detokenize.py`:581 passed, 251 skipped。测试文件跑在 "Async Engine, Inputs, Utils, Worker" CI 步。
- 风险:能跨 token 改写文本的 decoder(如多字符 `Replace`)理论上可能不同——检查过的 tokenizer 无一如此;prompt 以 U+FFFD 结尾时保持全 prompt seed,成本仍 O(prompt)(#48854 会把这类请求路由到慢 detokenizer)。

## 关联

- Supersedes #51281 与 #57330;相关预存在问题 #48853/#48854;fastokens shim 下(`VLLM_USE_FASTOKENS=1`, fastokens 0.3.6)`gpt2` 与 `Qwen3-0.6B` 测试通过。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
