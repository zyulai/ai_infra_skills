# PR #37505: [Fix] DP attention: correct the decode->extend prefix off-by-one

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37505](https://github.com/sgl-project/sglang/pull/37505) |
| 作者 | mmangkad |
| 合并日期 | 2026-09-02 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
DP attention 下 decode 转 extend 的 prefix 差一:RoPE 位置越过本行 KV 槽、causal 窗口多一个 token,改由 seq_lens_cpu 推导 prefix 修复。

## 问题现象
在 `--enable-dp-attention` 下,decode 步骤可能在「越过该行自身 KV 槽一位」的 RoPE 位置上计算 token,且 causal 窗口宽了一个 token;受影响步骤上所有在途请求的生成都会被腐蚀。触发条件:某次迭代在请求 decode 中途跑了 prefill,使 overlap 的 output_ids 滞后被排空——因此并发负载持续到达时很常见,空闲服务器上罕见。512 并发下的插桩显示约 31% 的转换请求(1226/3986)命中。正文给出 GSM8K(1319 例,temperature 0,Qwen3-30B-A3B-Instruct-2507,4x GB300,--tp-size 4 + --dp-size 4 --enable-dp-attention):main 分支 DP attention 仅 71.34/70.20/70.43,加本 PR 恢复到 94.24/94.47/94.39;关闭转换、回退 #36288 均恢复到约 94.2。

## 根因
正文明确披露:#36288(`ff5578eb4e`)加入的 DP decode->extend 转换中,`convert_decode_to_extend` 用请求的 token 列表推导 extend prefix:

```python
delta = 0 if self.enable_overlap else -1
self.prefix_lens = [len(r.origin_input_ids) + len(r.output_ids) + delta for r in self.reqs]
```

该公式假设 overlap 下 `output_ids` 总是比 `seq_lens` 滞后一步,实际并非如此:`seq_lens` 只在执行 `prepare_for_decode` 的迭代推进,而 pending result 无论哪次迭代都会落地,跑 prefill 的迭代会把滞后排空。1-token extend 的位置是 `arange(prefix_len, prefix_len + extend_len)`,需要 `prefix_len == seq_len - 1`;滞后被排空后旧公式给出 `prefix_len == seq_len`,即差一。DP-padding 路径早已用 `extend_prefix_lens = self.seq_lens - 1` 的正确约定,本 PR 让 scheduler 侧保持一致。

## 修复方案
- `python/sglang/srt/managers/schedule_batch.py` `convert_decode_to_extend`:改为从 `seq_lens_cpu` 推导 `prefix_lens` 与 `extend_range`(`set_extend_range(seq_len - 1, seq_len)`),并按 `bs` 切片(beam tail 行没有对应 req)、用 `zip(strict=True)` 让任何错位直接报错。
- `scheduler_components/dp_attn.py` `_local_prefill_cuda_graph_vote`:拒绝 beam 请求(转换后的 batch 走 prefill result path,会对 beam 请求走 per-req `commit_prefill` 而非 batch 级 `commit_decode`,且成员行没有自己的 req):`and all(r.beam_group is None for r in local_batch.reqs)`。
- `mix_with_running`:同一约定、同一位置改从 `seq_lens_cpu`/`req.seqlen - 1` 推导(正文注明这是加固而非实测修复,gsm8k over `--enable-mixed-chunk` 前后无差异)。

```diff
-        delta = 0 if self.enable_overlap else -1
-        self.prefix_lens = [
-            len(r.origin_input_ids) + len(r.output_ids) + delta for r in self.reqs
-        ]
+        seq_lens = self.seq_lens_cpu[:bs].tolist()
+        for req, seq_len in zip(self.reqs, seq_lens, strict=True):
+            req._refresh_fill_ids()
+            req.set_extend_range(seq_len - 1, seq_len)
+        self.prefix_lens = [seq_len - 1 for seq_len in seq_lens]
```

另新增 CPU 单测覆盖滞后存在/排空两种几何状态、beam 对齐与 strict-zip 失败路径。

## 影响与回归风险
- 影响窗口:#36288(`ff5578eb4e`)引入 DP decode->extend 转换之后、本 PR 之前。
- 回归面:影响 `--enable-dp-attention` 下 decode 转 extend 的调度路径(`convert_decode_to_extend`、`mix_with_running`)与 prefill cuda graph 投票门控;速度上正文实测该转换约值 2.2x(本 PR 15.1s vs 关闭转换 34.2s),beam 请求被排除在转换投票之外,spec 分支 `mix_with_running` 保持位对位等价的旧 `delta = -1` 语义。

## 关联
- https://github.com/sgl-project/sglang/pull/36288 (引入转换的 PR,bisect 定位)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
