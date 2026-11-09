# PR #33907: [Bugfix] Fix Random Dataset Prefix Length Inaccuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#33907](https://github.com/vllm-project/vllm/pull/33907) |
| 作者 | frankwang28 |
| 合并日期 | 2026-02-13 |
| labels | bug, performance, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
`RandomDataset` 的共享前缀此前直接采样 token id、不经 decode-encode 校正,实际前缀长度偏离 `--random-prefix-len`(4096/1024 配置下命中率 83.7% 而非期望 80%);让前缀走与 prompt 相同的 decode-encode 调整。

## 问题现象
benchmark 场景:`RandomDataset` 生成的共享前缀 token 长度不准确,与 `--random-prefix-len` 指定值存在偏差,影响 prefix cache 命中率基准的可信度。正文测试证据:`--random-prefix-len 4096 --random-input-len 1024` 时期望前缀命中率约 80%(4096/(4096+1024)=0.8),但修复前用 Qwen/Qwen3-8B 跑 `vllm bench throughput`(2048 prompts、`--enable-prefix-caching`)实测 Prefix cache hit rate 稳定在 83.7%-84.0%;修复后落到 79.9%-80.2%。

## 根因
正文说明:前缀生成本应像 input prompt 一样,把共享前缀 token 长度经 decode-encode tokenization 调整到目标长度,但 `RandomDataset` 的 `get_prefix` 未做这一步。
(由 diff 推断)旧 `get_prefix` 只是从 allowed_tokens 中按 `prefix_len` 随机采样 token id 直接返回;而 token 序列 decode 成文本再 encode 回来后长度常与目标不一致(`gen_prompt_decode_to_target_len` docstring 所述的采样不完美问题),prompt 生成路径已用该函数校正,前缀路径没有,于是实际共享前缀长度系统性偏离设定值。

## 修复方案
改 `vllm/benchmarks/datasets.py`:
1. `gen_prompt_decode_to_target_len` 返回值从 `(prompt, tokens)` 扩为 `(prompt, tokens, token_mismatch)`,重试预算耗尽时报告 `final_len - target_token_len`;
2. `get_prefix` 新增 `tokenizer` 参数:采样 `prefix_len` 个 token 后调用 `gen_prompt_decode_to_target_len(tokenizer=..., token_sequence=prefix_tokens, target_token_len=prefix_len, add_special_tokens=False, rng=self._rng)` 校正,`token_mismatch != 0` 时打 warning(说明这是采样流程不完美所致),返回 `adjusted_tokens`;
3. 两处 `sample()` 调用点改为传入 tokenizer。

```diff
-        prefix_token_ids = self.get_prefix(allowed_tokens, prefix_len)
+        prefix_token_ids = self.get_prefix(tokenizer, allowed_tokens, prefix_len)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`RandomDataset`(及同文件使用 `get_prefix` 的数据集类)的基准数据生成。前缀长度从「名义长度」变为「decode-encode 校正后的实际长度」,依赖旧行为做对比的历史基准数字不再可直接对比;`gen_prompt_decode_to_target_len` 返回值元数变化会影响其他调用点;校正失败时新增 warning 日志,数据不再静默偏差。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
