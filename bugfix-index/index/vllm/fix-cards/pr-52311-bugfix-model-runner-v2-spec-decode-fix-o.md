# PR #52311: [Bugfix][Model Runner V2][Spec Decode] Fix off-by-one in bad_words draft-prefix matching

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52311](https://github.com/vllm-project/vllm/pull/52311) |
| 作者 | jyan-R |
| 合并日期 | 2026-08-16 |
| labels | bug, mrv2, ready |
| 热度 | 👍 3 · 💬 12 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
MRV2 `_bad_words_kernel` spec 分支读 draft 区少了 `+1`,整体后移一位(boundary token 读两次);补上偏移使与兄弟 kernel 布局一致。

## 问题现象
bad_words 与投机解码组合、多 token bad word 的 prefix 窗口伸入 draft tokens 时:以 draft 区结尾的 prefix 匹配被漏掉(被禁 token 可被采样/接受);跨越 committed/draft 边界的匹配晚一行 verification row 才触发;重复读 boundary token 可能虚假匹配、误禁无辜 token。单 token bad word 与非 spec 行不受影响;Model Runner V1 是独立实现,不受影响。

## 根因
正文披露:sampler 传入的 `input_ids` 是按 `logits_indices` gathered 的,spec-decode 请求的 per-request 局部布局为:局部位置 0 = 最后一个已提交的采样 token,局部位置 j = draft token d_{j-1}。兄弟 kernel 都按此布局索引 draft(`_penalties_kernel` 用 `start_idx + prev_pos + 1`;thinking budget 的 `_load_effective_token` 用 `+ 1` 且有显式注释),而 `_bad_words_kernel` 读 `input_ids_ptr + cur_req_first_pos + spec_offset` 时**少了 `+ 1`**,每个 draft 区读取整体后移一个 token(boundary token 被读两次)。该 bug 自 kernel 引入(#33433)即存在。

## 修复方案
`vllm/v1/worker/gpu/sample/bad_words.py` 的 `_bad_words_kernel` spec 分支一行修正:load 地址加上 `+ 1`,与 penalties/thinking-budget kernel 的布局对齐。关键 diff 摘录:
```python
         from_spec_input = actual_pos >= output_len
         if from_spec_input:
+            # input_ids at local position 0 is the last committed token;
+            # draft tokens start at local position 1.
             spec_offset = actual_pos - output_len
-            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)
+            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset + 1)
```
新增首个覆盖该 kernel 的单测 `tests/v1/worker/test_gpu_bad_words.py`(committed `[10, 11]` + draft `[12, 13]`):边界跨越 prefix `[11, 12, 30]` 必须在 prefix 完成的那一行 mask;draft 内匹配 `[12, 13, 40]`;boundary token 防重复计数 `[11, 11, 50]`;committed-only 基线 `[10, 11, 60]`。

## 影响与回归风险
- 影响窗口:自 #33433 引入该 kernel 起至修复;影响 Model Runner V2 + bad_words + spec-decode 组合。
- 回归面:仅 `_bad_words_kernel` 的 spec 分支一行;非 spec 路径与 V1 实现不动。验证:A100 SXM4 80GB(torch 2.13.0+cu130、triton 3.7.1、nightly wheel `66728feb1`)上修复后 `4 passed`;换回修复前 kernel 则 `3 failed, 1 passed`(恰好三个 spec 分支用例失败)。交叉验证:kernel 标量循环转录为纯 Python,与独立参考匹配器跑 20,000 个随机 spec-decode 场景——带 `+ 1` 全部一致,不带约 24% 场景发散。

## 关联
- #33433(引入该 kernel)
- #34213(最接近的既有改动,仅涉 penalties,非重复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
