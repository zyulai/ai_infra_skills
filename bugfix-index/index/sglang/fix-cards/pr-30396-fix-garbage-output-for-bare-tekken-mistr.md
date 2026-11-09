# PR #30396: Fix garbage output for bare-tekken Mistral checkpoints (e.g. Leanstral)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30396](https://github.com/sgl-project/sglang/pull/30396) |
| 作者 | JustinTong0323 |
| 合并日期 | 2026-07-09 |
| labels | run-ci, bypass-fastfail |
| 热度 | 👍 1 · 💬 14 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
只带 `tekken.json` 的 Mistral checkpoint 经 HF 在线转换时 token id 整体偏移 1000,输出纯乱码;检测后改走 mistral-common 分词器修复。

## 问题现象
serving `mistralai/Leanstral-1.5-119B-A6B`(以及任何只带 `tekken.json`、没有 `tokenizer.json` 的 Mistral-native checkpoint)在所有硬件/attention 后端/TP 配置下输出纯乱码,同样权重在 vLLM 上正常。修复前 `"The capital of France is"` 生成 `" ÐµÑĢ äº ada ew ption ..."`;修复后输出 `" Paris. Paris is the largest city in France..."`,GSM8K(sgl-eval,全量 1319,temp 1.0 / top_p 0.95)accuracy 91.13%、stop rate 99.92%、0 错误,与 vLLM 行为一致。

## 根因
PR 正文披露:根因在分词器而非模型。`AutoTokenizer` 通过 transformers 的 `TekkenConverter` 在线转换 `tekken.json`,BPE id 从 rank 0 开始分配,**丢掉了 tekken id 空间中位于 BPE 词表之前的 1000 个 special-token 槽位**,导致每个编码 id 偏移 1000:

```
text: "The capital of France is"
mistral-common (reference): [1784, 8961, 1307, 5498, 1395]
converted HF tokenizer:     [784, 112623, 1935, 37166, 275]
```

id < 1000 落到未训练的 special-token embedding 行(全零),其余落到错误行——garbage in, garbage out。正文并指出:即使修正偏移,转换后的分词器切词也与 mistral-common 不同,转换路径不可挽救。

## 修复方案
1. `python/sglang/srt/utils/hf_transformers/mistral_utils.py`:新增 `is_bare_tekken_checkpoint()`——checkpoint 有 `tekken.json` 且无 `tokenizer.json` 即为真(本地目录或 HF cache,不联网;冷缓存下 tokenizer 先于权重加载,tekken.json 尚未缓存时,用 HEAD 请求探测 hub 判定,任一出错都回落默认路径)。
2. `python/sglang/srt/utils/hf_transformers/common.py`:新增辅助 `_cached_file_exists()` 与 `_remote_file_exists()`。
3. `python/sglang/srt/utils/hf_transformers/tokenizer.py` `get_tokenizer`:检测命中时改用 `transformers.tokenization_mistral_common.MistralCommonTokenizer`(mistral-common 后端,`mistral_common` 已是硬依赖),结果照旧经 `_apply_post_load_fixes`;带官方 `tokenizer.json` 的 checkpoint 不受影响。关键路由:

```python
if is_bare_tekken_checkpoint(tokenizer_name, tokenizer_revision):
    tokenizer = MistralCommonTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision
    )
else:
    tokenizer = _auto_tokenizer_from_pretrained(...)
```

4. 新增单测 `test/registered/unit/tokenizer/test_tekken_tokenizer_routing.py`:检测逻辑 + 用真实 Leanstral `tekken.json` 与 mistral-common 参考 id 做端到端 encode 一致性(2 passed)。

## 影响与回归风险
- 影响窗口:正文未说明;受影响面为所有仅携带 `tekken.json` 的 Mistral-native checkpoint 部署。
- 回归面:tokenizer 加载路径新增一个探测分支,含本地目录检查、HF cache 探测,以及冷缓存时的 hub HEAD 探测(离线/网络异常/受限 repo 下均返回 False 走原路径)。命中检测的 checkpoint 忽略 `tokenizer_backend` 配置改走 mistral-common;其余 checkpoint 逻辑不变。探测在启动期执行,增加少量文件存在性检查开销。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
