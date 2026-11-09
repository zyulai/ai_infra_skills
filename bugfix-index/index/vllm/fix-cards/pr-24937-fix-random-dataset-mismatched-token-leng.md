# PR #24937: Fix random dataset mismatched token length with config.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24937](https://github.com/vllm-project/vllm/pull/24937) |
| 作者 | weireweire |
| 合并日期 | 2025-09-28 |
| labels | performance, ready |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
random 数据集 decode→encode 不保长,目标 8192 实得 8179/8185;改为迭代重试逼近目标长度,残余偏差汇总告警。

## 问题现象
按正文测试命令运行 `vllm bench serve --backend vllm ... --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0`(模型 gpt-oss-120b):修复前打印长度可见 `len(token_sequence), target_token_len: 8179, 8192`、`8185, 8192` 等不匹配情况;修复后全部为 8192。

## 根因
正文明确披露:有时 `seq != tokenizer.decode(tokenizer.encode(seq))`,一次 decode 再 re-encode 不能保证 token 长度不变,需要迭代修正直到匹配。diff 中新函数的 docstring 举例说明该非 1:1 现象:GPT2Tokenizer 下 `[6880, 6881] -> ['Ġcalls', 'here'] -> [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']`——相邻 token 合并/拆分导致 re-encode 后长度漂移;旧实现只做一次 decode→encode→截断,补齐逻辑(递归生成 additional consistent tokens)也不迭代收敛。

## 修复方案
`vllm/benchmarks/datasets.py`:
- 新增 `gen_prompt_decode_to_target_len(tokenizer, token_sequence, target_token_len, max_retry=10, ...)`:decode 后 re-encode,长度不足用 rng 补随机 token(`0 ~ vocab_size`)、超出则截断,循环至匹配或重试耗尽,返回 `(prompt, adjusted_token_sequence, token_mismatch)`;
- `RandomDataset.generate_token_sequence` 与 `RandomMultiModalDataset` 改用该函数,`total_input_len = len(adjusted_token_sequence)`;各 dataset 的 `sample` 累计 `token_mismatch_total`,非零时以 `logger.warning` 汇总报告(说明这是采样过程不完美所致);
- `_generate_exact_length_tokens`(prefix 生成)同样改用该函数并返回 mismatch,返回签名从单值变为 `(tokens, mismatch)`。

## 影响与回归风险
- 回归面:仅 `vllm bench serve` 的 random / random-multimodal 数据集生成路径(基准工具,不影响推理服务);修复后 prompt 长度更贴近 `--random-input-len`,但 token_mismatch 仍可能非零(有告警提示);`_generate_exact_length_tokens` 返回签名变化属内部接口;prefix+suffix 拼接场景中 mismatch 现按 prefix/suffix 分别累计统计。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
