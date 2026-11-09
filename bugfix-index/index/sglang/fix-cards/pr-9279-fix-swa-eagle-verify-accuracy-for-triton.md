# PR #9279: Fix swa eagle verify accuracy for Triton backend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9279](https://github.com/sgl-project/sglang/pull/9279) |
| 作者 | ispobock |
| 合并日期 | 2025-08-17 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
Triton attention 后端在 SWA 层 + EAGLE verify 时 custom mask 读取缺少滑窗内偏移,verify 精度受损;把 window_kv_offsets 传入 kernel 修正 mask 索引,MMLU 0.777 提到 0.816。

## 问题现象
EAGLE(实测用 EAGLE3)+ sliding window 模型(gpt-oss-20b)在 `--attention-backend triton` 下 verify 阶段精度下降:复现命令为 `--speculative-algorithm EAGLE3 --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --attention-backend triton`;MMLU(simple-evals,1000 examples)o4-mini 主线 0.777,本 PR 后 0.816。正文未给出更多现象细节。

## 根因
正文披露:"对 sliding window 层,应该加上 offset 去读取滑动窗口内的 custom mask"(原文 "For sliding window layers, we should add offset to read custom mask in the sliding window"),即 SWA 下按全量序列铺开的 custom mask 索引与窗口内 kv 位置错位。

## 修复方案
- `triton_backend.py`:`ForwardMetadata` 增加 `window_kv_offsets`;`update_sliding_window_buffer(_cuda_graph)` 返回值增加 `window_kv_start_idx`;decode/extend 元数据构建及 cuda graph capture/replay 全链路透传(新增 `cuda_graph_window_kv_offsets` 缓冲,并对 `seq_lens[:bs]` 切片);`forward_extend` 把 `window_kv_offsets` 传给 `extend_attention_fwd`。
- `triton_ops/extend_attention.py`:`_fwd_kernel` 新增 `window_kv_offset_ptr`,`USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0` 时按序列加载偏移,custom mask 索引改为含 `window_kv_offset` 的行跨度与列偏移;`extend_attention_fwd` 签名透传。
- `triton_ops/decode_attention.py`:两处 batch 维由 `kv_indptr.shape[0] - 1` 改为 `q.shape[0]`。

关键 diff(extend kernel 的 mask 索引):
```diff
             custom_mask = tl.load(
                 mask_ptr
                 + cur_seq_mask_start_idx
-                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
+                + (cur_block_m * BLOCK_M + offs_m[:, None])
+                * (cur_seq_len + window_kv_offset)
+                + window_kv_offset
+                + start_n
+                + offs_n[None, :],
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Triton attention 后端的 SWA + custom mask(EAGLE verify)路径与 cuda graph capture/replay;decode_attention kernel 的 batch 维推导改动影响所有 triton decode 调用(由 diff 推断:q 行数与 kv_indptr 段数在窗口/非窗口下应一致);非 SWA 或无 custom mask 时 window_kv_offset 为 0,索引行为不变。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
