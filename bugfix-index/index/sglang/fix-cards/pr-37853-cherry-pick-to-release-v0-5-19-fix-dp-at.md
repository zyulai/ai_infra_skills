# PR #37853: [Cherry-pick to release/v0.5.19] [Fix] DP attention: correct the decode->extend prefix off-by-one (#37505)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37853](https://github.com/sgl-project/sglang/pull/37853) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2026-09-03 |
| labels | bypass-maintenance, cherry-pick |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
DP attention 下 decode 转 extend 时 prefix 推导差一位,overlap lag 排空后 RoPE 位置越过本行 KV 槽位;改为用 seq_len-1 作为 prefix。

## 问题现象
正文为 cherry-pick 说明(源 PR #37505、commit `718bd39fe08b42e0095daa571510e47c40c588d2` cherry-pick 到 `release/v0.5.19`),未描述具体现象,仅标题指出 DP attention decode->extend 存在 prefix off-by-one。新增单测 docstring 描述:dp decode->extend 视图必须把新 token 放在 `seq_len - 1`,否则 RoPE 位置会落在该行自身 KV 槽位之后一位(由 diff 推断)。

## 根因
正文未披露根因(仅标题点出 off-by-one)。由 diff 推断:原实现用 `len(req.full_untruncated_fill_ids)` 或 `len(origin_input_ids) + len(output_ids) + delta` 推 prefix;对 overlap 调度器 output_ids 有一步延迟,当该延迟被排空(例如某轮跑了 prefill)时,该公式会得到 `prefix_len == seq_len`,即 extend 位置越过该行自己的 KV 槽位一位;beam tail 行还会使 reqs 对齐的列表长度错配。diff 注释同时说明:与 `convert_decode_to_extend` 相同的不变量是「caller 已跑过 prepare_for_decode,tail 的 prefix 等于行长度 - 1」。

## 修复方案
- `python/sglang/srt/managers/schedule_batch.py`:
  - `mix_with_running`:按 `spec_algorithm` 分支取 `running_prefix_lens`(非 spec 用 `seq_lens_cpu - 1`,spec 用 `r.seqlen - 1`),`req.set_extend_range(prefix_len, prefix_len + 1)`;`prefix_lens` 直接复用 `running_prefix_lens`,删除 output_ids delta 补偿逻辑。
  - `convert_decode_to_extend`:用 `self.seq_lens_cpu[:bs].tolist()` 逐行取 `seq_len`,`set_extend_range(seq_len - 1, seq_len)`,`prefix_lens = [seq_len - 1]`,并用 `zip(strict=True)` 使行/req 不一致时报错。
- `python/sglang/srt/managers/scheduler_components/dp_attn.py` `_local_prefill_cuda_graph_vote`:新增 `all(r.beam_group is None for r in local_batch.reqs)` 条件,beam 请求不再投赞成票走转换后的 prefill result path。
- 新增单测 `test_schedule_batch_convert_decode_to_extend.py`(覆盖 overlap lag 存在/排空、beam tail、短 seq_lens 报错)并更新 `test_schedule_batch_out_of_place.py`。

```diff
-            full_len = len(req.full_untruncated_fill_ids)
-            req.set_extend_range(full_len - 1, full_len)
+            req.set_extend_range(seq_len - 1, seq_len)
```

## 影响与回归风险
- 影响窗口:cherry-pick 目标为 release/v0.5.19 分支;正文未说明更早的引入版本。
- 回归面:DP attention 下 decode->extend 转换与 `mix_with_running` 的 extend_range/prefix_lens 计算(含重叠与非重叠调度,现统一为 seq_len-1);beam search 请求在 prefill CUDA graph 投票路径(beam 请求被排除,不再走 prefill result path);投机解码的 spec 分支 prefix 改用 `r.seqlen - 1`。

## 关联
- https://github.com/sgl-project/sglang/pull/37505 (源 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
