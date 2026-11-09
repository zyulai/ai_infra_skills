# PR #46399: [Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers instead of crashing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46399](https://github.com/vllm-project/vllm/pull/46399) |
| 作者 | Sunt-ing |
| 合并日期 | 2026-10-17 |
| labels | bug, speculative-decoding, ready, v1 |
| 热度 | 👍 1 · 💬 20 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
ngram_gpu/suffix/medusa/extract_hidden_states 不支持动态 SD,运行时 K 低于配置即断言崩溃;config 初始化时禁用动态 SD 并回退静态 K。

## 问题现象
启用 dynamic speculative decoding(`num_speculative_tokens_per_batch_size`,#32374 加入)配置 `ngram_gpu`/`suffix`/`medusa`/`extract_hidden_states` 时,scheduler 可合法选择低于最大值的运行时 K(batch size 1 时可到 0),proposer 在 `LLM.generate()` 深处抛 `AssertionError`——此时配置已被接受、target model 已经跑过。E2E 复现(RTX 4090,`Qwen/Qwen3-0.6B`,V1 model runner,schedule `[(1, 1, 0), (2, 8, 4)]`):`ngram_proposer_gpu.py:341` 的 `assert num_speculative_tokens == self.k` 与 `suffix_decoding.py:44` 的 `assert num_speculative_tokens == self.num_speculative_tokens` 崩溃;修复后同一调用跑完(如 `"ha ha ha ha ha ha ha ha" -> " ha ha ha ha"`,batch size 2 的 runtime K=2 也正常)。

## 根因
正文披露:这四个 fixed-K proposer 写于 dynamic SD 特性之前,硬断言运行时 K 等于配置的最大值,不支持动态降低 K。支持该特性的 proposer 会遵循运行时 K(eagle family 在 `llm_base_proposer.py` 覆写 `self.num_speculative_tokens`;CPU ngram 用 `assert num_speculative_tokens <= self.k`)。

## 修复方案
`vllm/config/vllm.py` 新增 `_maybe_disable_dynamic_sd_for_v1()`:V1 model runner 下,若方法不在 `("ngram", "eagle", "eagle3", "mtp", "draft_model", "dflash")` 且启用动态 SD,则 `warning_once` 并把 `num_speculative_tokens_per_batch_size` 置 None,回退静态 `num_speculative_tokens`;沿用 #45963 数据并行场景的既有模式(`_maybe_disable_dynamic_sd_for_data_parallel`),在其之后调用。回归测试 `test_dynamic_sd_disabled_for_fixed_k_method`(参数化:ngram_gpu 禁用、ngram 不变)在 main 上失败、本 PR 通过;`tests/v1/spec_decode/test_dynamic_sd.py` 全文件 18 passed。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围(动态 SD 自 #32374 引入)。
- 回归面:V1 speculative config 初始化;支持动态 SD 的方法(eagle family、CPU ngram)行为不变(测试含负对照);被禁用动态 SD 的配置从运行时崩溃变为「告警 + 静态 K」。

## 关联
- #32374(dynamic speculative decoding 特性来源)
- #45963(data-parallel 场景禁用动态 SD 的既有模式)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
