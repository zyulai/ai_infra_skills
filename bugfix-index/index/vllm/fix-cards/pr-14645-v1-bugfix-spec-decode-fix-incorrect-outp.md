# PR #14645: [V1][Bugfix][Spec Decode] Fix incorrect outputs in V1 speculative decoding due to batch indexing

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#14645](https://github.com/vllm-project/vllm/pull/14645) |
| 作者 | benchislett |
| 合并日期 | 2025-03-12 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
V1 投机解码用 `num_scheduled_tokens` 的字典键序推断 batch 顺序,与 `input_batch.req_ids` 不一致导致输出串包;改为按 `req_ids` 索引。

## 问题现象
V1 投机解码(ngram)在批内多个请求具有不同投机长度(speculative lengths)时输出错误:输出概率切片到各投机解码请求的分组依赖顺序假设,一个请求的补全里会「泄漏」其他请求的 token。原有 E2E 测试未能捕捉该问题,因为 batch size 不够大、请求之间过于相似。作者更新 E2E 测试为 100 条混合请求(一部分可被 n-gram 完整预测、一部分不能),在 main 分支上运行出现大量失败。修复后通过启发式断言(≥70% 精确匹配)。

## 根因
正文披露:`gpu_model_runner.py` 中 batch 内请求按 `self.input_batch.req_ids` 排序;而投机解码代码却从字典 `scheduler_output.num_scheduled_tokens` 的键来推断 batch 顺序,该字典的键序并不保证与 `req_ids` 一致。当批内请求投机长度不同时,输出概率切片的分组就依赖这个顺序,于是错位。

## 修复方案
一行修复,`vllm/v1/worker/gpu_model_runner.py` 的 `execute_model`:draft_token_ids 的推导从迭代 `scheduled_request_ids`(即 `scheduler_output.num_scheduled_tokens.keys()`)改为迭代 `self.input_batch.req_ids`:

```python
draft_token_ids = [
    scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
    for req_id in self.input_batch.req_ids
]
```

同时重写 `tests/v1/e2e/test_ngram_spec_decode.py`:随机生成 100 条 repeat/sentence 混合 prompt(固定种子 0),对比开关投机解码的输出,断言至少 70% 精确匹配。

## 影响与回归风险
- 影响窗口:正文未写明版本范围(问题在 V1 引擎,`VLLM_USE_V1=1`)。
- 回归面:运行时改动仅影响 V1 GPU worker 的投机解码路径(rejection sampler 的 draft_token_ids 与 target_probs 对齐方式);非投机解码路径不变。测试改动放宽为启发式 70% 匹配阈值而非逐条精确相等,可能降低对其他 ngram 回归的敏感度。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
