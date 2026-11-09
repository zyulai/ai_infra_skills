# PR #4134: Fix a draft model accuracy bug in eagle; support step=1; return logprob in eagle

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4134](https://github.com/sgl-project/sglang/pull/4134) |
| 作者 | merrymercy |
| 合并日期 | 2025-03-06 |
| labels | (none) |
| 热度 | 👍 5 · 💬 1 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论

EAGLE `prepare_for_extend` 循环漏写 `pt += extend_len`,各请求的 draft 输入互相覆盖造成 accuracy bug;同时支持 speculative_num_steps=1、为 eagle 补上输出 logprob 计算。

## 问题现象

正文描述极简,列了三件事:draft model accuracy bug(根因行是 `pt += extend_len` 缺失)、支持 step=1、返回 logprobs。未给出具体报错或现象描述。由 diff 的配套测试改动看,修复后 gsm8k 的 `avg_spec_accept_length` 断言阈值从 > 2.9 提到 > 3.5、e2e 延迟阈值从 1100ms 收紧到 900ms,可间接印证此前 draft 精度受损、accept length 偏低。(由 diff 推断)

## 根因

正文披露:accuracy bug 的根因是 `eagle_utils.py` `prepare_for_extend` 中缺少 `pt += extend_len` 这一行。机制(由 diff 推断):该函数按请求循环把 verified_id 拼接写入 `batch.input_ids[pt : pt + extend_len]`,但循环内 pt 从不推进,导致除第一个请求外,后续请求都覆写同一区间——送入 draft model extend 的输入错乱,精度随之受损。step=1 不可用则是 `build_tree_kernel_efficient_preprocess` 对 `parents_list` 长度为 1 的情形直接 `torch.cat(parents_list[:-1], dim=1)` 拼接空列表;logprob 缺失是 eagle verify 路径从未计算/挂载输出 logprobs。

## 修复方案

- accuracy:`prepare_for_extend` 循环末尾补 `pt += extend_len`(正文点名的根因行)。
- step=1:`build_eagle_tree.py` 中当 `len(parents_list) == 1` 时构造 `(batch_size, 0)` 空的 `parent_list`,不再对空列表做 cat。
- logprob:`eagle_worker.py` `verify()` 中在 `batch.return_logprob` 时用 target sampler 的 `update_output_logprobs` 计算输出 logprobs(按 `accept_length_per_req + 1` 计每请求 token 数,含 bonus token),逐请求追加 `output_token_logprobs` 与 top logprobs;`forward_draft_extend(_after_decode)` 显式置 `return_logprob = False` 以省掉 draft 侧开销。
- 配套重构/优化:flashinfer backend 增加 tensor-core decode 的 `plan` 路径与 `global_override_indptr_cpu` 快通道、`draft_extend_cuda_graph_metadata`;`init_forward_metadata_replay_cuda_graph` 显式传 `bs`;CUDA graph capture 移除 `torch.cuda.synchronize()` 与 TP barrier;EAGLEWorker 初始化拆出 `init_attention_backend()`。

关键摘录(eagle_utils.py):

```diff
             batch.input_ids[pt : pt + extend_len] = torch.concat(
                 (input_ids[1:], self.verified_id[i].reshape(1))
             )
+            pt += extend_len
```

测试:新增 `test_logprob_start_len`、`test_logprob_match`、`test_logprob_mixed` 三个 eagle logprob 测试;调整 accept length / 延迟阈值;CI matrix 增加 48-100 分片。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,涉及所有 EAGLE speculative decoding 用户——draft 输入构造、树构建(step=1 新路径)、verify 后的 logprob 输出路径、draft/flashinfer attention backend 的 plan 与 CUDA graph capture 流程(移除 capture 中的同步 barrier 属行为变化)、draft worker 初始化顺序。step>1 的树构建与非 logprob 请求的主路径逻辑不变。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
