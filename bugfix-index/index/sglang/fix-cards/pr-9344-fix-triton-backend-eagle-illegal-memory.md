# PR #9344: Fix triton backend eagle illegal memory access

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9344](https://github.com/sgl-project/sglang/pull/9344) |
| 作者 | ispobock |
| 合并日期 | 2025-08-20 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
EAGLE3 + triton attention 下 extend/decode kernel 的 `kv_indices * stride` 发生 int32 溢出导致 illegal memory access;把 kv_indices 相关缓冲改为 int64 修复。

## 问题现象
使用 triton attention backend 的 EAGLE3 投机采样出现 CUDA illegal memory access(extend 与 decode kernel 均涉及)。正文给出的复现命令:

```
python3 -m sglang.launch_server --model openai/gpt-oss-20b --speculative-algorithm EAGLE3 --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --attention-backend triton
```

随后用 lm_eval 跑 gsm8k(128 并发、max_gen_toks=2048)可触发;修复后 exact_match flexible-extract 0.8961、strict-match 0.3601。

## 根因
正文披露:`kv_indices * stride` 可能溢出——kv_indices 以 int32 分配,EAGLE 投机路径下索引规模放大(`batch_size * topk * max_context_len` 等),索引乘 stride 后超出 int32 范围,访问越界显存。

## 修复方案
将 `python/sglang/srt/layers/attention/triton_backend.py` 中所有 kv_indices 类缓冲的 dtype 从 `torch.int32` 统一改为 `torch.int64`,覆盖:normal/extend 路径的 `kv_indices` 分配(spec、prefix 两个分支,另有一处 `kv_indices.to(torch.int64)`)、CUDA graph 的 `cuda_graph_kv_indices`、`cuda_graph_window_kv_indices`、EAGLE 投机路径的索引缓冲,以及 `update_sliding_window_buffer` 的 `window_kv_indices`。kernel 本身不改,依赖 Triton 对指针类型的推导。关键 diff 摘录:

```diff
             kv_indices = torch.empty(
-                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
+                forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
             )
```

```diff
             self.cuda_graph_kv_indices = torch.zeros(
                 (self.speculative_num_steps, max_num_tokens * self.max_context_len),
-                dtype=torch.int32,
+                dtype=torch.int64,
                 device=self.device,
             )
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围。
- 回归面:仅 `triton_backend.py` 的索引缓冲 dtype,涉及普通 batch、extend、EAGLE 投机、CUDA graph、sliding-window 各路径。int64 使索引缓冲显存翻倍(`max_num_tokens * max_context_len` 量级),超大上下文配置下显存占用略增;Triton kernel 侧按传入指针类型编译,已注册的 CUDA graph 需重建。非 triton 后端与非 EAGLE 场景逻辑不变。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
