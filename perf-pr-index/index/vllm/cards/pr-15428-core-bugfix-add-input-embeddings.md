# PR #15428: [Core] [Bugfix] Add Input Embeddings

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qthequartermasterman |
| 合并 | 2025-05-02 |
| 分类 | Other（tier: major） |
| Label | frontend, speculative-decoding, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「10 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/15428 |

## 一句话结论

支持把 `prompt_embeds` 传给 `LLM.generate`，使仅微调 embedding 层的场景可用；PR 未给出性能数字。

## 动机

PR 正文说明该改动是为了支持"只微调 embedding 层"的用例，并让同一模型后端能够承载多套自定义调优过的 embedding 层；正文列出 FIX #416、FIX #8323、FIX #14621 三个 issue。NOTE 中说明本 PR 只是 #11684 在 main 上 rebase 并修正 pre-commit 错误的版本（因 @Bryce1010 已久未更新原 PR）。正文未提及性能瓶颈或吞吐目标，标题为 `[Core] [Bugfix]`，属输入接口的功能补齐。

## 优化点

改动沿输入管线铺开：在 `vllm/inputs/data.py` 新增 `EmbedsPrompt`（含 `prompt_embeds: torch.Tensor`）与 `EmbedsInputs`（`type="embeds"`）两个 TypedDict 及 `embeds_inputs()` 构造函数，并把 `SingletonPrompt`、`SingletonInputs`、`DecoderOnlyInputs` 的 Union 扩展到 embeds 类型；`parse.py`、`preprocess.py`、`__init__.py` 相应支持解析与预处理 embeds 输入。`vllm/core/scheduler.py` 的 `_schedule_prefills` 增加 `using_prompt_embeds` 判断，禁止把使用 prompt_embeds 与不使用的 sequence group 混进同一 batch；`_schedule_default` 也会从 decode 批次里剔除类型不一致的 group。引擎侧 `llm_engine.py`、`async_llm_engine.py`、`output_processor/single_step.py`、`multi_step.py` 及 flashinfer 后端随之适配。这是接口与调度约束的改动，未涉及内核或算法优化。

## 关键改动

```text
vllm/inputs/data.py: 新增 EmbedsPrompt / EmbedsInputs 类型与 embeds_inputs()，扩展输入 Union
vllm/core/scheduler.py: 组批时禁止 prompt_embeds 与 token 输入混排，decode 批次过滤不一致 group
vllm/inputs/parse.py: 解析 embeds 形式 prompt
vllm/inputs/preprocess.py: 预处理 embeds 输入
tests/core/test_scheduler.py: 新增 test_no_batches_mixed_with_prompt_tokens_and_prompt_embeds
```

```python
            # We cannot mix sequence groups that use prompt embeds and
            # those that do not.
            if len(seq_groups) == 0:
                using_prompt_embeds = seq_group.uses_prompt_embeds()
            if using_prompt_embeds != seq_group.uses_prompt_embeds():
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
```

## 性能声明

PR 正文未给出定量性能数据（pack 的 perf claims 检测为 none）；PR 未注明测试条件。正文只描述功能用例与 FIX 的 issue，没有任何吞吐、延迟或显存类 benchmark 数字，本卡无法给出性能结论。

## 验证与回归风险

diff 中新增 `tests/core/test_scheduler.py::test_no_batches_mixed_with_prompt_tokens_and_prompt_embeds`，验证 scheduler 不会把 token 与 embeds 输入混排；`tests/models/language/generation/test_common.py` 在 `VLLM_USE_V1=0` 时用 HF `get_input_embeddings()` 由 token ids 生成 prompt_embeds，并用 `check_logprobs_close` 对比 token 路径与 embeds 路径的输出一致性；`tests/worker/test_model_runner.py` 亦有改动。可能影响输入解析与预处理（data/parse/preprocess/__init__）、scheduler 组批、`llm_engine` / `async_llm_engine`、output_processor 与 flashinfer 后端；回归重点是两类输入不得混批、且生成结果需对齐。pack 末尾注明 10 file(s) dropped from diff，完整改动以 PR 为准。正文未给出 Test Result 章节。

## 关联

正文明确说明本 PR 是 PR #11684 的 rebase 版（修 pre-commit 错误），并列出 FIX #416、#8323、#14621 三个 issue；未声明其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
