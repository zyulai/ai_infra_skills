# PR #15790: [MTP][spec_v2] Fix TRTLLM MLA backend crash in EAGLE draft_extend mode

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15790](https://github.com/sgl-project/sglang/pull/15790) |
| 作者 | YAMY1234 |
| 合并日期 | 2026-01-10 |
| labels | blackwell, run-ci |
| 热度 | 👍 1 · 💬 10 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论

trtllm_mla 后端在 EAGLE draft_extend 模式下 accept_length 不齐导致 reshape 崩溃;增加 can_direct_view/pad-unpad 处理并修正 seq_lens_q 为 accept_length+1。

## 问题现象

TRTLLM MLA 后端在 EAGLE 投机解码 `draft_extend` 模式、各序列 `accept_length` 不同时崩溃。作者给出 agg 复现命令(`SGLANG_ENABLE_SPEC_V2=1 ... --speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-num-draft-tokens 3 --attention-backend trtllm_mla` 等),并提到 disagg(PD 分离)设置中的间歇性问题。报错位于 `trtllm_mla_backend.py:1046 forward_extend`:

```
RuntimeError: shape '[9, -1, 32, 576]' is invalid for input of size 516096
```

## 根因

正文披露:draft_extend 模式下各序列的 `accept_length` 不同,`forward_extend` 未处理这种变长情况(修改说明 1:增加 `can_direct_view` 检查,在 `total_tokens % bs != 0` 时做 padding/unpadding)。由错误信息可见 `q.view(bs, -1, tp_q_head_num, head_dim)` 在总 token 数不能被 batch size 整除时形状非法(如 516096 无法按 `[9, -1, 32, 576]` 切分)。(由 diff 推断)此外 `init_forward_metadata_replay_cuda_graph` 中 `cu_seqlens_q`/`seq_lens_q` 按裸 `accept_length` 累计——每序列 query token 实为 `accept_length + 1`,少算 1 导致元数据与实际 q 布局不一致。

## 修复方案

改 `python/sglang/srt/layers/attention/trtllm_mla_backend.py`:

1. `forward_extend`(draft_extend 分支):`total_tokens % bs == 0` 时直接按 `tokens_per_seq` reshape(`can_direct_view`);否则用 `forward_batch.extend_seq_lens(_cpu)` 的真实值算 `actual_max_seq_len_q`/`actual_cu_seqlens_q`,把 q pad 到 `(bs, max_seq_len_q, ...)`(优先复用 `self.padded_q_buffer`,否则新建零张量),经 `pad_draft_extend_query` 填充后跑 kernel,输出再经 `unpad_draft_extend_output` 去填充。target_verify 分支维持整批相同 draft token 数的直接 view。
2. `init_forward_metadata_replay_cuda_graph`:`seq_lens_q`/`cu_seqlens_q` 的计算改用 `accept_length + 1`(正文修改说明 2):

```diff
+            # draft_extend uses (accept_length + 1) query tokens per sequence
+            extend_seq_lens = accept_length + 1
             metadata.cu_seqlens_q[1:].copy_(
-                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
+                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
             )
-            metadata.seq_lens_q.copy_(accept_length)
+            metadata.seq_lens_q.copy_(extend_seq_lens)
```

## 影响与回归风险

- 回归面:trtllm_mla 后端的 EAGLE `draft_extend` 与 `target_verify` 前向路径及 CUDA graph replay 元数据;target_verify 行为不变,仅 draft_extend 变长分支新增 pad/unpad(引入额外拷贝,新路径走 padded_q_buffer 复用)。作者实测 longbench_v2 得分 0.601;disagg 基准中 Accept Length 1.3→2.41、Accept Rate 约 43%→约 80%。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
