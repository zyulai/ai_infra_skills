# PR #5765: [Speculative Decoding] Enabling bonus token in speculative decoding for KV cache based models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sroy745 |
| 合并 | 2024-07-10 |
| 分类 | Speculative Decoding (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 展示部分含 `multi_step_worker.py`、`spec_decode_worker.py` 相关测试等,另有 6 个文件因体积截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/5765 |

## 一句话结论

V0 投机解码让 KV cache 模型的 bonus token 进入 MultiStepWorker 视野(batch 扩展),e2e 提速约 33%。

## 动机

issue #4212:V0 投机解码中,target 模型每次 forward 会在接受 token 后额外采样一个 bonus token,但 MultiStepWorker 的 drafter 多步推进并不知道上一轮哪些 sequence 拿到了 bonus token,导致 drafter 的 KV cache 状态与 target 不一致,后续提议建立在错误上下文上。这限制了 KV cache 型 drafter(如 MultiStepWorker)的投机解码正确性/效率。

## 优化点

三步机制。其一,`SpecDecodeWorker` 维护 `_seq_with_bonus_token_in_last_step` 集合:每次 `_create_output_sampler_list` 根据 accepted_token_ids 的最后一位(=-1 表示无 bonus)刷新该集合,并维护 `_request_id_seq_id_mapping` 以便在 sequence/request 结束时清理(finished_requests_ids 处理)。其二,`MultiStepWorker.sampler_output` 在 step 0 前做 batch 扩展:`_expand_execute_model_request` 为每个含 bonus token 的 sequence 额外构造一个"去掉最后一个 token"的副本 sequence(`_copy_seq_metadata_excluding_last_token`,同步回退 num_computed_tokens),扩展后的 batch 一起前向;完成后 `_filter_model_output` 只保留原 sequence 的输出。这样 drafter 每一步都能从正确(不含 bonus)的上下文继续提议。其三,`SpeculativeProposer` 接口与 medusa/ngram/mlp 等 worker 全部加 `seq_ids_with_bonus_token_in_last_step` 参数(多数为 unused),另加 `--disable-bonus-tokens-in-kv-cache` 开关。

## 关键改动

```text
vllm/spec_decode/multi_step_worker.py: _expand_execute_model_request、
  _copy_seq_metadata_excluding_last_token、_filter_model_output
vllm/spec_decode/spec_decode_worker.py: _seq_with_bonus_token_in_last_step
  维护与清理(文件在截断部分)
vllm/sequence.py: get_all_seq_ids_and_request_ids 辅助函数
vllm/spec_decode/interfaces.py 与各 worker: 接口参数传递
tests/spec_decode/test_multi_step_worker.py 等三个测试文件
```

## 性能声明

e2e 测试(不使用 cuda graphs):draft 模型 JackFram/llama-68m,target JackFram/llama-160m,batch size 64。num_speculation=1 时 completion time 约 33% 提速:w/o bonus 64 prompts 10.13 it/s(est. output 2592.40 toks/s)→ with bonus 15.50 it/s(est. output 3968.28 toks/s)。PR 未注明 GPU 型号等其他条件。

## 验证与回归风险

新增测试:`test_multi_step_with_batch_expansion_correct_output`(正确指定 bonus 集合时输出与单步一致)、`_incorrect_output`(错误指定时仅正确子集输出匹配)、`test_populate_seq_ids_with_bonus_tokens`(状态刷新语义)、`test_handle_finished_requests`(结束请求清理状态);既有 ngram/medusa/multi_step 测试同步更新接口签名。风险:PR 自述"sequence 终止时未清空集合,需要通知机制"(后续以 finished_requests_ids 处理);batch 扩展增大 drafter 前向 batch(每个含 bonus 的 sequence 翻倍),依赖 KV 空间充足(`_assert_enough_kv_space` 保留);仅 MultiStepWorker 利用 KV cache,其他 worker 传参不变。

## 关联

修复 issue #4212。V0 投机解码系列;同分类后续演进:#13365(V1 N-gram)、#22872/#24505/#25916(V1 EAGLE)、#23569(V1 异步执行与投机解码不兼容问题)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
