# PR #22963: [Structured Outputs] [Bug] Fix misalignment in apply_grammar_bitmask causing unintended masking and NaN logits

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#22963](https://github.com/vllm-project/vllm/pull/22963) |
| 作者 | rishitdholakia13 |
| 合并日期 | 2025-08-15 |
| labels | ready, v1 |
| 热度 | 👍 3 · 💬 8 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
apply_grammar_bitmask 用预分配 sorted_bitmask 的长度决定是否传索引,与实际需掩码请求错位,非结构化请求 logits 被 NaN 化;改为按 out_indices 判断。

## 问题现象
在 v1 的 `apply_grammar_bitmask` 路径上,结构化输出与非结构化输出请求混合调度时,非结构化输出请求也被 grammar bitmask 掩码,其 logits 被转换成 NaN,进而引发 illegal memory access 问题。

## 根因
正文披露:bitmask 的数量是从预初始化的 `sorted_bitmask` 张量推导,而不是从实际需要掩码的索引推导。当二者不一致(bitmask 行数恰与 logits 行数相同但实际需要对齐的请求索引不同)时产生错位:kernel 误认为 bitmask 已与 logits 对齐而跳过传索引,把不该掩码的请求也掩掉了。具体机制上,原代码用 `grammar_bitmask.shape[0] == logits.shape[0]` 决定 `skip_out_indices`;batch 中含非结构化请求时 `sorted_bitmask` 的行数与真实待掩码的 `out_indices` 数不再对应(由 diff 推断)。

## 修复方案
改 `vllm/v1/worker/gpu_model_runner.py` 的 `apply_grammar_bitmask` 一行判断:

```python
-        skip_out_indices = grammar_bitmask.shape[0] == logits.shape[0]
+        skip_out_indices = len(out_indices) == logits.shape[0]
```

即从「bitmask 行数 == logits 行数」改为「实际需要掩码的 out_indices 数 == logits 行数」,确保只有 bitmask 与 logits 真正逐行对齐时才跳过索引参数,掩码只作用于相关请求。正文未附测试计划与结果(Test Plan/Test Result 留空)。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围(labels 标注 v1,即 v1 引擎的该函数引入后的范围)。
- 回归面:仅 `apply_grammar_bitmask` 中是否向 kernel 传 `out_indices` 的判断;纯结构化输出 batch(两者相等)行为不变,混合 batch 下掩码不再外溢到非结构化请求。

## 关联
正文未列出关联 issue/PR(仅 cc @russellb @aarnphm)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
