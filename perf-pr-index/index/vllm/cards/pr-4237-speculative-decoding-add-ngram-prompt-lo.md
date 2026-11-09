# PR #4237: [Speculative decoding] Add ngram prompt lookup decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @leiwen83 |
| 合并 | 2024-05-01 |
| 分类 | Speculative Decoding（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「7 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/4237 |

## 一句话结论

新增 ngram prompt lookup 投机解码，直接从 prompt 取 draft、无需额外 draft 模型，正文无性能数字。

## 动机

PR 正文很短：算法细节指向 HuggingFace 的 assisted-generation 博客，代码直接参考 transformers 的当时实现（transformers#27775）。核心论点是"Since we directly get draft from prompt, there is no need another model or modified model to get the proposal"——既然 draft 直接来自 prompt，就不需要另一个模型或改造过的模型来产生 proposal，因此这是享受 speculative decoding 加速最便捷的方式（"the most convenient way to enjoy the speedup of speculation"）。正文未提及具体瓶颈、模型、硬件或任何量化目标。

## 优化点

新增 `NGramWorker`（继承 `LoraNotSupportedWorkerBase`），实现 prompt lookup decoding：对每条序列，从 `min(ngram_prompt_lookup_max, input_length - 1)` 递减到 `ngram_prompt_lookup_min` 尝试 ngram_size，取末尾 `ngram_tensor`，用 `input_ids.unfold` 生成所有窗口并做 `(windows == ngram_tensor).all(dim=1)` 匹配；找到多于一个匹配时，取匹配位置之后的一段 token 作为 draft proposal（不足 `sample_len` 时补 0），再用 `scatter_` 构造 `sampled_token_probs` / `sampled_token_ids` 交给 proposer。`NGramWorker` 不执行模型、不占 KV cache：`execute_model`、`determine_num_available_blocks`、`initialize_cache` 均为 pass，`get_cache_block_size_bytes()` 返回 0，注释称其提供"a light drafter without need for model"。配套把 proposer 接口通用化——测试与调用侧由 `DraftModelTop1Proposer` 改为 `Top1Proposer`、`execute_model_multi_step(num_steps=...)` 改为 `sampler_output(sample_len=...)`、`max_proposal_len` 改为 `proposal_len`。配置侧 `maybe_create_spec_config` 新增 `ngram_prompt_lookup_max` / `ngram_prompt_lookup_min` 参数，当 `speculative_model == "[ngram]"` 时走 ngram 分支（要求 max > 0 且 max > min，min 缺省为 0）；`arg_utils.py` 新增 `--ngram-prompt-lookup-max` / `--ngram-prompt-lookup-min` 两个 CLI 参数。

## 关键改动

```text
vllm/spec_decode/ngram_worker.py: 新增 NGramWorker，实现 ngram prompt lookup 匹配生成 draft proposal
vllm/config.py: maybe_create_spec_config 支持 [ngram] 与 ngram_prompt_lookup_max/min 参数及校验
vllm/engine/arg_utils.py: 新增 --ngram-prompt-lookup-max / --ngram-prompt-lookup-min CLI 参数
vllm/spec_decode/multi_step_worker.py、batch_expansion.py、vllm/executor/gpu_executor.py: 配合 proposer 接口调整
tests/spec_decode/test_ngram_worker.py: 新增 ngram 算法正确性测试（单条无匹配、批量部分/全部匹配等）
```

```python
            for ngram_size in range(
                    min(self.ngram_prompt_lookup_max, input_length - 1),
                    self.ngram_prompt_lookup_min,
                    -1,
            ):
                ngram_tensor = input_ids[-1 * ngram_size:]
                windows = input_ids.unfold(dimension=0,
                                           size=ngram_size,
                                           step=1)
                matches = (windows == ngram_tensor).all(dim=1)
                match_indices = matches.nonzero(as_tuple=True)[0]
                if match_indices.size()[0] > 1:
                    has_spec_out = True
```

## 性能声明

PR 正文未给出定量性能数据（pack 的 perf claims 检测为 none）；PR 未注明测试条件。正文只有定性的"享受 speculative decoding 加速最便捷的方式"这一表述，没有任何吞吐、延迟、接受率或加速比数字，也没有模型、硬件、精度、workload 条件。本卡无法给出量化性能结论。

## 验证与回归风险

diff 中新增 `tests/spec_decode/test_ngram_worker.py`（约 206 行，含 `test_ngram_algo_correctness_for_single_no_match`、`test_ngram_algo_correctness_for_batches_not_match_all`、`test_ngram_algo_correctness_for_batches_match_all` 等，用 `create_worker` 与 `set_ngram_window_size(0, 3)` 验证候选匹配），并修改 `tests/spec_decode/test_multi_step_worker.py`（`DraftModelTop1Proposer`→`Top1Proposer`、`execute_model_multi_step`→`sampler_output`、`max_proposal_len`→`proposal_len`）。可能影响的路径：speculative decoding 的 proposer 接口与 `multi_step_worker` / `batch_expansion` / `gpu_executor` 调用链、`SpeculativeConfig` 构造（`[ngram]` 分支的参数校验）、以及依赖原 `DraftModelTop1Proposer` 名称的测试。正文未给出 Test Plan / Test Result 章节；pack 末尾注明 7 file(s) dropped from diff，完整改动以 PR 为准。

## 关联

无（正文仅引用 HuggingFace assisted-generation 博客与 transformers PR #27775 作为算法与代码来源，未指向 vLLM 内的前序/后续 PR）

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
