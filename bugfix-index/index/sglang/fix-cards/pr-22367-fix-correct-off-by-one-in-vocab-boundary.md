# PR #22367: fix: correct off-by-one in vocab boundary check for token validation

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22367](https://github.com/sgl-project/sglang/pull/22367) |
| 作者 | Ricardo-M-L |
| 合并日期 | 2026-06-09 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

词表越界检查 off-by-one 漏判 token_id == vocab_size,把 `>` 改为 `>=` 对齐 spec_utils 的正确边界。

## 问题现象

`schedule_batch.py` 的 `_check_vocab_boundary_finish` 负责检测越界 token id(例如 logits 出现 NaN 或采样错误产生的非法 id)。此前检查条件漏掉了 `token_id == vocab_size` 这一非法值,使这类损坏输出不会被 NaN guard 拦截、请求不会按预期 finish。正文未给出线上具体故障日志,主要从边界语义层面描述影响。

## 根因

正文披露:合法 token id 的取值范围是 `[0, vocab_size)`,即 `token_id == vocab_size` 已经越界;而原判断 `token_id > self.vocab_size` 只拦截严格大于的值,漏掉了等于的情况。同文件其他位置的 `clamp_(0, self.model_config.vocab_size - 1)` 也印证了合法上界是 `vocab_size - 1`;`speculative/spec_utils.py`(line 597)的同类检查用的是正确的 `>=`:

```python
if vocab_size and curr_token_id >= vocab_size:
    accepted = False
```

## 修复方案

`python/sglang/srt/managers/schedule_batch.py` 的 `_check_vocab_boundary_finish` 中,把判断条件由 `token_id > self.vocab_size or token_id < 0` 改为 `token_id >= self.vocab_size or token_id < 0`,触发后逻辑不变(把越界槽位改写为 eos/stop token 并以 "NaN happened" 结束请求)。

关键 diff 摘录:

```diff
     def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
         for i, token_id in enumerate(new_accepted_tokens):
-            if token_id > self.vocab_size or token_id < 0:
+            if token_id >= self.vocab_size or token_id < 0:
```

新增回归单测 `test/registered/unit/managers/test_vocab_boundary_finish.py`,覆盖 `token_id == vocab_size` 必须触发、`vocab_size - 1` 不触发、负数 id 触发、无 eos 时改写为 stop token 等边界。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `_check_vocab_boundary_finish` 一处比较符;合法请求(全部 id 在 `[0, vocab_size)`)行为完全不变,只有恰好生成 `vocab_size` 这个越界 id 的请求会从"漏判"变为"正确触发 finish"。与 `spec_utils.py` 的既有判定保持一致,风险很小。CI 状态栏显示 Extra 测试为失败状态。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
