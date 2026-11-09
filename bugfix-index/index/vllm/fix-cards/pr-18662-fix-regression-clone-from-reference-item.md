# PR #18662: fix(regression): clone from reference items

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18662](https://github.com/vllm-project/vllm/pull/18662) |
| 作者 | aarnphm |
| 合并日期 | 2025-05-24 |
| labels | ready |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
V1 并行采样在 add_request 循环里克隆的是上轮已改写的 params 而非原始参数,副本相互污染;改为从 original_params 克隆。

## 问题现象
V1 测试持续失败(consistent failure),场景为 add_request 循环中对解引用的 original_params 做 clone 的并行采样(parallel sampling)路径,对应 issue #18656。正文未附具体报错日志。

## 根因
正文披露:失败源于 add_request 循环中克隆的是被解引用(dereference)的 params,而不是原始的 original_params。(由 diff 推断)在 `for i in range(original_params.n)` 循环里,`params = params.clone()` 使第一轮之后每轮克隆的都已是上一轮被改写过的对象(`n` 已被置 1、`seed` 已偏移),各并行采样副本继承并继续污染彼此的参数,而非各自从原始参数出发。

## 修复方案
`vllm/sequence.py` 的 `add_request` 中,parallel sampling 分支把克隆源从被修改的 `params` 换回 `original_params`,保证每个副本都从原始参数出发:

```diff
         for i in range(original_params.n):
             request_id_i = f"{request_id}_parallel_sample_{i}"
             group.seq_id_to_index[request_id_i] = i
-            params = params.clone()
+            params = original_params.clone()
             params.n = 1
             if params.seed is not None:
                 params.seed += i
```

## 影响与回归风险
- 影响窗口:正文未说明(该克隆逻辑存在于 V1 的 add_request 并行采样路径)。
- 回归面:仅 `vllm/sequence.py` `add_request` 中 parallel sampling(n>1)分支的一行克隆源改动;`params.n = 1` 与 `params.seed += i` 的后续逻辑保持不变。克隆源改为原始参数后,每轮克隆相互独立,n 与 seed 的语义与直觉一致,非并行采样(n=1,不进循环)路径不受影响。

## 关联
- #18656(被修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
