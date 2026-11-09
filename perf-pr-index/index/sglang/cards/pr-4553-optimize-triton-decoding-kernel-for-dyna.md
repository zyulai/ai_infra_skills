# PR #4553: Optimize Triton decoding kernel for dynamic workload

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Alcanderian |
| 合并 | 2025-03-19 |
| 分类 | Attention / Kernel (tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与 changed files 数;diff 本身可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/4553 |

## 一句话结论

把 Triton decode attention 的静态 KV split 改为按 SM 数、seq_lens、head 数与 batch 动态调度 num_kv_splits,128 并发 decode 吞吐 +10% 以上并超过 flashinfer backend。

## 动机

原 Triton decode attention 使用静态 KV split(`triton_attention_num_kv_splits` 对所有 batch 统一生效,实际每 batch 都拿满 8 个 split),无法适应在线服务中动态变化的 context 长度与 batch size,导致性能退化——大 batch 短上下文时每个请求仍被切成 8 份,split 碎片化浪费 SM 与带宽。

## 优化点

三层机制改动:(1) split 数调度——新增 Triton kernel `get_num_kv_splits_triton`,在 GPU 上一次性为整个 batch 算出 per-batch `num_kv_splits`:用 max/min seq_lens 的比值定 `kv_chunk_size_1`,再按 `device_core_count`(随 seqlen 对数放大)与 `bh_grid`(bs x 每组 head 数,GQA 时 block_h 取 min(16, kv_group))定 `kv_chunk_size_2`,每 batch 取两者 cdiv 的较大者,上限 `max_kv_splits`;可用 `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=1` 或 bs>4096 时退回静态满 split。(2) logits 布局——`attn_out` 拆成 `attn_logits`(v_head_dim 对齐)与 `attn_lse` 两个张量,stage1 把 `e_max + log(e_sum)` 写入独立 LSE 张量而非原来塞在 `v_head_dim+1` 最后一列。(3) 分块粒度——`kv_chunk_size` 对齐 32(`_MIN_BLOCK_KV`),`kv_len_per_split = cdiv(cdiv(seq_len, kv_splits), 32) * 32`,限制短上下文的实际 split 数;stage1 grid 用 `max_kv_splits` 启动但每 split 先算自己的起止,超出范围直接空跑;stage2 按 per-batch `kv_splits` 循环归约。接口上 `init_forward_metadata_replay_cuda_graph` 增加 `num_kv_heads` 参数,CUDA graph 路径经 `cuda_graph_num_kv_splits` 缓冲在 replay 时重算 split。q 的加载也移入 `split_kv_end > split_kv_start` 分支避免无效加载。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py: stage1/stage2 接受 per-batch num_kv_splits 与 MIN_BLOCK_KV=32 对齐;attn_logits/attn_lse 分离
python/sglang/srt/layers/attention/triton_backend.py: 新增 get_num_kv_splits_triton kernel 与 get_num_kv_splits 调度;num_kv_splits 进 forward_metadata
python/sglang/srt/layers/attention/base_attn_backend.py 及各 backend: init_forward_metadata_replay_cuda_graph 签名加 num_kv_heads
python/sglang/srt/model_executor/cuda_graph_runner.py: replay 时从 kv pool 推断 num_kv_heads 并传给 attention backend
test/srt/test_triton_attention_kernels.py: 测试改为 (attn_logits, attn_lse) 二元组与 per-batch num_kv_splits
```

关键片段(动态 split 调度核心):

```python
max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
ext_device_core_count = device_core_count * tl.maximum(
    tl.cast(tl.ceil(tl.log2(ext_seq_len)), tl.int32), 1)
max_kv_splits_2 = tl.minimum(tl.cdiv(ext_device_core_count, bh_grid), max_kv_splits)
num_kv_splits = tl.maximum(
    tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2))
```

## 性能声明

简单基准(decode 日志吞吐,未注明模型/硬件):

- 128 并发:实际 num_kv_splits 从 8 降到 1,gen throughput 12940.23 -> 14403.42 token/s(短上下文 #token 24477)与 5662.88 -> 6234.16 token/s(长上下文 #token 275357),+10% 以上,且高于 flashinfer backend 同条件数字(14296.91 / 5802.22)
- 1 并发:实际 num_kv_splits 保持 8,吞吐不变(155.01 -> 155.05 / 141.59 -> 141.34 token/s);flashinfer 对照为 148.51 / 148.80
- `triton_attention_num_kv_splits` 改为 16 时 1 并发 155.18 / 145.84 token/s

CI offline 基准(`test_offline_throughput_with_triton_attention_backend`,500 requests):Total token throughput 10566.40 -> 10893.63 tok/s,Mean E2E 84579.50 -> 82093.30 ms,Mean ITL 36.92 -> 35.64 ms,PR 自述为"slightly improvement"。

## 验证与回归风险

更新 `test/srt/test_triton_attention_kernels.py` 的 decode 与 grouped decode 单测以匹配新接口(attn_logits/attn_lse 元组、per-batch num_kv_splits);Checklist 中"Add unit tests"未勾选。回归面:所有使用 Triton attention backend decode 的模型;CUDA graph capture/replay 签名变化波及 flashinfer、flashinfer_mla 等 backend(传 -1 占位)与 DPO分层 backend;`get_num_kv_splits_triton` 自注 "this method is tunable" 且对 seq_lens 长度悬殊时用 max/10 近似,调度策略本身可调;HIP 路径 BLOCK=8 等特判保留。

## 关联

直接前序:#1138(引入 grouped decode kernel,本 PR 在其 stage1/stage2 上加动态 split);同文件后续优化可参考同 batch 其他卡。pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
