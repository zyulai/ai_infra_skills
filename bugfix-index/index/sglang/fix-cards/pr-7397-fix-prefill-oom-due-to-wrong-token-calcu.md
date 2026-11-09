# PR #7397: Fix prefill OOM due to wrong token calculation when page > 1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7397](https://github.com/sgl-project/sglang/pull/7397) |
| 作者 | hnyls2002 |
| 合并日期 | 2025-06-23 |
| labels | (none) |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
更新 prefill 预算时未把 extend_input_len 对齐到 page size,page>1 时预计算不足致 prefill OOM,现统一向上取整。

## 问题现象
prefill OOM。触发条件:page_size > 1,且处于 chunked prefill 或 ignore eos 模式(正文原文:"we do not align the `extend_input_len` to the page size, which can happen in chunked prefill and ignore eos mode")。正文未给出具体报错日志。

## 根因
正文披露:更新 prefill budget 时未将 `extend_input_len` 按 page size 对齐,预算扣减按未取整的 token 数计算,而实际 KV cache 分配按页对齐,导致预算高估可用 token,page > 1 时在 chunked prefill 与 ignore eos 场景下实际占用超出预算,引发 OOM。

## 修复方案
改 `python/sglang/srt/managers/schedule_policy.py`:
1. 新增 `ceil_paged_tokens(tokens)`,把 token 数向上取整到 page size 的整数倍;`_prefill_one_req` 重命名为 `_update_prefill_budget`,函数内先对 `extend_input_len` 做 page 对齐再扣减预算:

```diff
-    def _prefill_one_req(
+    def _update_prefill_budget(
         self, prefix_len: int, extend_input_len: int, max_new_tokens: int
     ):
+        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
+        extend_input_len = self.ceil_paged_tokens(extend_input_len)
+
         self.rem_total_token_offset += extend_input_len + max_new_tokens
```

2. `add_one_req_ignore_eos`:入口先检查取整后的输入 token 是否超过剩余预算,不足直接返回 `AddReqResult.NO_TOKEN`;并把 ignore eos 的剩余 token 估算改为保守(预留 `IGNORE_EOS_RESERVE_TOKENS = 1` × bs)。
3. chunked prefill 与普通 `add_one_req` 的各处预算更新统一改用 `_update_prefill_budget`,原先两处手写的 `-(-x // page_size) * page_size` 收敛为 `ceil_paged_tokens`。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;问题见 #7353。
- 回归面:调度器 prefill 预算计算(`schedule_policy.py`)。所有 page_size>1 的 chunked prefill 与 ignore eos 请求预算扣减变大(页对齐取整),调度更保守,可能轻微降低打包密度换取不 OOM;ignore eos 模式新增早退与预留 token 逻辑,作者自己留 TODO 注明该 workaround"可能过于保守"。

## 关联
- [#7353](https://github.com/sgl-project/sglang/issues/7353)(本 PR 声明修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
