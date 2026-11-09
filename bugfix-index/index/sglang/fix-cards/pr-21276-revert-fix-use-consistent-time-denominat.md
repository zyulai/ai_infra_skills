# PR #21276: Revert "fix: use consistent time denominator for throughput metrics in bench_one_batch_server"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21276](https://github.com/sgl-project/sglang/pull/21276) |
| 作者 | ch-wan |
| 合并日期 | 2026-03-24 |
| labels | (none) |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | regression |

## 一句话结论

回滚 #19223:`bench_one_batch_server` 吞吐指标分母改回原口径(input 用 last_ttft、output 用 latency - last_ttft),撤销统一为总 latency 的改动。

## 问题现象

正文未描述具体现象。正文仅一句 "Reverts sgl-project/sglang#19223",未填写 Motivation/Modifications,也未给出复现与影响面描述。

## 根因

正文未披露根因。(由 diff 推断)被回滚的 #19223 曾把 `run_one_case` 中的吞吐计算分母统一改为总 `latency`(`batch_size * input_len / latency` 与 `batch_size * output_len / latency`);本 revert 恢复原来的口径,即 input 吞吐除以 `last_ttft`、output 吞吐除以 `latency - last_ttft`,使指标回到 prefill/decode 分段统计的语义。

## 修复方案

按 revert 提交恢复 `python/sglang/test/bench_one_batch_server_internal.py` 中 `run_one_case` 的指标计算:

```diff
-    input_throughput = batch_size * input_len / latency
-    output_throughput = batch_size * output_len / latency
+    input_throughput = batch_size * input_len / last_ttft
+    output_throughput = batch_size * output_len / (latency - last_ttft)
     overall_throughput = batch_size * (input_len + output_len) / latency
```

仅此一处改动,`overall_throughput` 的分母保持不变。

## 影响与回归风险

- 影响窗口:含 #19223 改动的版本受影响(依据正文 revert 声明;正文未给出具体版本号)。
- 回归面:改动仅触及内部基准脚本 `bench_one_batch_server_internal.py` 的吞吐打印口径,不影响服务端推理路径;恢复后 input/output throughput 数值含义与 #19223 之前一致,依赖新口径的对比数据或脚本需要重新对齐。

## 关联

- [#19223](https://github.com/sgl-project/sglang/pull/19223)(正文声明被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
