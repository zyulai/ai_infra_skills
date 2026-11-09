# PR #40589: [Bugfix] Fix dp mtp hang

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#40589](https://github.com/vllm-project/vllm/pull/40589) |
| 作者 | SherryC41 |
| 合并日期 | 2026-07-06 |
| labels | bug, ready, v1 |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
DP+MTP 下个别 rank 因输入超出 drafter 限制跳过 draft,其他 rank 仍执行 drafter 前向,集合通信不一致导致 hang;在跳过分支补 dummy_run。

## 问题现象
作者用 DeepSeek-R1 跑 bench:dp 4 tp 2 mtp 3,`--max-model-len 4100`,出现 hang。触发条件是某个 DP rank 的 `proposed_daft_token_ids` 被跳过——max_seq_len(4098) + num_spec_token(3) > max_model_len(4100) 使 `_input_fits_in_drafter` 判为不 fit——而其他 DP rank 仍在执行 `drafter.dummy_run()`。测试命令为 `vllm serve /models/DeepSeek-R1 ... -dp 4 -tp 2 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` 配合 `vllm bench serve`(random 数据集,input 3072 / output 1024,32 并发)。原实现必现 hang,修复后可跑通。

## 根因
正文披露:hang 由「一个 DP rank 跳过 `proposed_daft_token_ids`,而其他 DP rank 仍执行 `drafter.dummy_run()`」的不一致引起。drafter 的前向(EAGLE/DraftModel 等)携带 TP/EP/DP 集合通信(diff 注释:「drafter_runs_model_forward ... thus carries TP/EP/DP collectives」),当各 rank 对 `input_fits_in_drafter` 判断不一致时,部分 rank 进入 drafter 前向、部分 rank 完全跳过,集合通信无法对齐而挂死(正文明确给出该触发机制,diff 中两处注释也写明 "Prevent hang when DP ranks disagree on input_fits_in_drafter")。

## 修复方案
改 `vllm/v1/worker/gpu_model_runner.py` 的 `propose_draft_token_ids` 路径:抽出 `drafter_runs_model_forward` 标志(与 padded-drafter-batch 开关解耦);在 padded 分支中,`input_fits_in_drafter` 为假且 `data_parallel_size > 1` 时补调 `self.drafter.dummy_run(num_tokens=1)`;在 bookkeeping 后的分支(变量更名为 `draft_after_bookkeeping`)同样补上,并断言 drafter 类型为 EagleProposer/DFlashProposer/DraftModelProposer/ExtractHiddenStatesProposer/Gemma4Proposer 之一。

```diff
+                    if self.parallel_config.data_parallel_size > 1:
+                        # Prevent hang when DP ranks disagree on input_fits_in_drafter
+                        self.drafter.dummy_run(num_tokens=1)
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`gpu_model_runner.py` 中 draft token 提议逻辑,影响所有使用 GPU drafter(EAGLE/DraftModel/ExtractHiddenStates 等)且 DP>1 的 speculative decoding 场景;新增的 `dummy_run(num_tokens=1)` 只在不 fit 且 DP>1 的分支执行,单 DP 或 ngram 等 CPU drafter 路径行为不变;`draft_after_bookkeeping` 由条件赋值改为恒真赋值,语义等价(原 False 分支走不到该执行点)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
