# PR #39833: [MRv2]fix: model accuracy regression caused by reusing the stale last_sampled_tokens and draft_tokens

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39833](https://github.com/vllm-project/vllm/pull/39833) |
| 作者 | liuzijing2014 |
| 合并日期 | 2026-04-21 |
| labels | bug, mrv2, ready, v1 |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
V2 model runner 复用请求状态槽位时,PD 分离场景首步 decode 读到上一请求残留的 `last_sampled_tokens`/`draft_tokens`,造成约 13% 的 GSM8K 精度回退;在 `add_request()` 中初始化这两个状态修复。

## 问题现象
PD 分离(prefill/decode disaggregation)场景下,V2 model runner 出现约 13% 的 GSM8K 精度回退:修复前 V2 = 0.82,V1 基线 = 0.93,修复后 V2 = 0.94。验证方式:在 Crusoe GB200 NVL72 上以 concurrency=32 跑完整 GSM8K 评测。触发条件是 decode 侧收到已完成 prefill 的请求后,首个 decode 步就读到被同槽位前一请求污染的状态。

## 根因
PR 正文披露:PD 分离下,decode 侧收到的请求带有 `num_computed_tokens == prefill_len`(prefill 已全部完成),第一个迭代就是 decode 步——没有 prefill 步通过 `postprocess()` 来初始化 `last_sampled_tokens`。此时 `combine_sampled_and_draft_tokens` 内核读取的是占用过同一槽位的前一个请求残留的旧值(stale value)。`draft_tokens` 同样存在跨请求泄漏(正文 Summary 指出两者在槽位复用时泄漏)。bug 位置:`vllm/v1/worker/gpu/states.py:add_request()`。

## 修复方案
在 `vllm/v1/worker/gpu/states.py` 的 `add_request()` 中,当 `num_computed_tokens > 0 and num_computed_tokens <= prefill_len`(PD 分离或恢复的请求)时,把 `last_sampled_tokens[req_idx]` 初始化为 `all_token_ids[num_computed_tokens - 1]`(最后一个已计算 token),保证首个 decode 步拿到正确的 input_id;并一律 `self.draft_tokens[req_idx].zero_()` 清空 draft tokens。关键 diff:

```diff
+        if num_computed_tokens > 0 and num_computed_tokens <= prefill_len:
+            self.last_sampled_tokens[req_idx : req_idx + 1] = all_token_ids[
+                num_computed_tokens - 1
+            ]
+        self.draft_tokens[req_idx].zero_()
```

代码注释说明:全新 prefill 请求(`num_computed_tokens == 0`)的张量不会被 `combine_sampled_and_draft_tokens` 读取,故跳过写入;使用切片赋值而非标量索引,使写入经由 `fill_` 派发、避免 host/device 同步。

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:仅改动 `states.py` 的 `add_request()`。影响所有 V2 model runner 的请求入队路径:PD 分离与断点恢复请求的首个 decode 步输入被纠正(即修复目标);全新 prefill 请求路径仅新增一次 `draft_tokens` 清零写,该张量此前可能残留旧值,清零属于保守修复。注释声称切片赋值不引入 host/device 同步,对每请求入队开销影响很小。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
