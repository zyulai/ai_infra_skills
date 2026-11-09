# PR #57454: [Bugfix][DSv4.1] Preserve NaN-scored candidate block indices

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57454](https://github.com/vllm-project/vllm/pull/57454) |
| 作者 | WoosukKwon |
| 合并日期 | 2026-09-18 |
| labels | bug, ready, DSv4.1 |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
NaN 得分的候选块被 `-inf` 过滤误删致前缀重复块,DeepGEMM 断言崩溃;改为仅排除 -inf padding。

## 问题现象
DeepSeek-V4.1 TP4 MegaAttention 在 dummy prefill/autotuning 阶段启动即失败,报 DeepGEMM sparse-slot 断言(捕获的失败行中出现 389 份 block 388 的拷贝)。触发条件:候选源 logits 含 NaN 时(暖机/ dummy run 常见)。有限分数场景行为不变。

## 根因
正文披露:候选源 logits 含 NaN 时,`torch.topk` 仍会选出这些块的 ID,但存储候选时的谓词 `value > -inf` 把 NaN 得分的块当作无效而丢弃,行内只剩下被固定的最新块;行未被填满,扩展该行时会把最后一个块重复填入,而前缀部分 DeepGEMM 要求块唯一,于是触发 sparse-slot 断言。NaN 本身源自 dummy run 的上游数值问题,本 PR 不做 sanitize、不修上游数值源。

## 修复方案
修改 `vllm/model_executor/kernels/attention/dsa/candidate_blocks.py` 的 `_store_candidates_kernel`:把"排除 padding"的判断从 `value > -float("inf")` 改为 `value != -float("inf")`,只排除真正的 `-inf` padding 哨兵,保留 NaN 得分块的 ID;测试参考实现同步改为 `top.values != -torch.inf`。不加 kernel/buffer/fallback/host 同步。

```diff
+    # NaN scores can occur during warmup; only -inf denotes padding.
     tl.store(
         output + row * out_stride_row + cols * out_stride_col,
-        tl.where(value > -float("inf"), index, -1),
+        tl.where(value != -float("inf"), index, -1),
         cols < OUT_K,
     )
```

## 影响与回归风险
- 影响窗口:正文未说明;涉及使用 candidate blocks 选择器(DSA/短 prefill indexer)且 logits 可能为 NaN 的配置。
- 回归面:仅候选存储谓词一处;`-inf` padding 仍被正确排除,有限分数输出与原实现逐位一致(8K/50K/100K 随机、零、NaN、混合 NaN 分数验证)。NaN 块被保留后依赖 DeepGEMM 接受唯一块——正文验证四组 TP-rank 输入经未修改 DeepGEMM 重放恢复 389 个唯一块;TP4 MegaAttention 真实权重启动 smoke 通过(短/长 prompt 返回预期答案),但未做完整精度评测。

## 关联
- PR #56558
- PR #50201
- PR #49897
- https://github.com/vllm-project/DeepGEMM(pull)#16
- 正文注明无关联 issue("No linked issue")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
