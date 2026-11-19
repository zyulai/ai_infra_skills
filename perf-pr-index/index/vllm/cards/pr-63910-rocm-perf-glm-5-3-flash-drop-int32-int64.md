# PR #63910: [ROCm][Perf][GLM-5.3-Flash] Drop int32/int64 copies in the kpool sparse indexer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-sriram |
| 合并 | 2026-11-18 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, ready, glm |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 tests/kernels/test_kpool_decode_update_batched.py、vllm/models/glm5next/amd/ops/kpool_compress.py、vllm/models/glm5next/amd/sparse_indexer.py |
| 链接 | https://github.com/vllm-project/vllm/pull/63910 |

## 一句话结论

GLM-5.3-Flash 的 ROCm kpool 稀疏索引器让 `expand_pools_and_append_tail` 直接消费 int32 pool id / int64 positions(`+1` 移入 kernel)并原地写满宽写入 `topk_indices_buffer`,每层稀疏注意力少 6 次 launch、每 decode 步少 66 次,MI355X TP4 上中位 TPOT 在 1k–128k 工作负载降 1.7–4.4%。

## 动机

kpool 稀疏索引器(`vllm/models/glm5next/amd/sparse_indexer.py`)在每层稀疏注意力都要围绕 `expand_pools_and_append_tail` 额外启动 int32↔int64 cast、一个 `+1`、一个 `-1` fill 和一次 buffer copy;trace 显示单个稀疏注意力层 6 个被移除的 launch 合计约 25 µs,每 decode 步这类 launch 达 70 个、kernel 时间 324 µs。既然 expand kernel 可以按原 dtype 读入并在内部做 `+1`、直接写 buffer(含 `-1` tail),这些中间拷贝就是纯开销。

## 优化点

1. `expand_pools_and_append_tail` 新签名:`seq_len_offset`(caller 可直接传 positions,seq_len = pos + 1)与 `out`(int32 `[rows, W]` 行切片原地写,`-1` 填到 `topk + pool_size - 1` 之后),输入 int32/int64 均可,不再需要 cast。
2. `_kpool_decode_update_batched_kernel` 直接接受 int64 positions,去掉 narrowing copy;pool ids 保持 int32(< 65k pools,`pid * pool_size` 不溢出)。
3. sparse_indexer 引入 `decode_rows_self_filled` 标记:decode 行由 kernel 写满宽后,`-1` fill 只覆盖其后的行;decode 快路径直接 `expand_pools_and_append_tail(..., out=topk_indices_buffer[:n])` 后返回。
4. prefill chunk 走同一直接写入路径。

## 关键改动

```python
# vllm/models/glm5next/amd/ops/kpool_compress.py(_expand_pools_and_append_tail_kernel)
-    seq_len = tl.load(seq_lens_ptr + row)
+    # SEQ_LEN_OFFSET=1: callers pass positions, seq_len = pos + 1.
+    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32) + SEQ_LEN_OFFSET
```

```python
# vllm/models/glm5next/amd/sparse_indexer.py(decode 行自填后,-1 fill 缩小范围)
-    topk_indices_buffer[: hidden_states.shape[0]] = -1
+    decode_rows_self_filled = (has_decode and index_kpool > 1 and positions is not None
+        and attn_metadata_narrowed.decode is not None
+        and not attn_metadata_narrowed.decode.requires_padding)
+    fill_start = num_decode_tokens if decode_rows_self_filled else 0
+    if fill_start < hidden_states.shape[0]:
+        topk_indices_buffer[fill_start : hidden_states.shape[0]] = -1
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- MI355X, TP4, `amd/GLM-5.3-Flash-Quark-MXFP4`,对同一 ROCm nightly:1.7–4.4% lower median TPOT at 1–32 concurrent requests on 1k/1k, 8k/1k, 64k/1k and 128k/1k;0.9–1.8% at 64;flat at 64k c16 (−0.2%);128k c8 为 +2.1%,在其 run-to-run spread 内(该 arm 两次 run 相差 5.7%)。
- MTP(multi-token decode,padded layout):median TPOT −1.6% geomean with MTP3(14 of 18 points faster)、−1.5% with MTP5(13 of 18 faster;worst +2.7% at 8k c4)。
- A decode step issues 66 fewer kernels (1,295 → 1,229) and 308 µs less kernel time;被移除的 6 个 launch(如 `vectorized_elementwise_kernel<FillFunctor<int>>` 5.5 µs)每次 trace 合计约 25 µs。
- gfx942(MI325X,`zai-org/GLM-5.3-Flash` FP8, TP4):total throughput +1.9% geomean, every one of 15 points faster (+0.5% to +3.5%);mean TPOT −2.0%。
- Accuracy:GSM8K 0.920 vs 0.921;GPQA-Diamond 0.904/0.909 vs 0.914/0.894;RULER NIAH(`niah_single_2`, 64k)在 TP2/TP4/TP8、MTP off/3/5 全部 1.0。MTP3 GSM8K 0.923→0.916,MTP5 0.918→0.922。

## 验证与回归风险

- 单测:`tests/kernels/test_kpool_decode_update_batched.py` 45 passed(34 个既有用例 + int64 positions 5 个(ROCm only)+ `test_expand_pools_and_append_tail_in_place` 6 个,校验 in-place 调用与 reference 一致且 slice 外行不动);`tests/v1/attention/test_kpool_tail_slot_mapping.py`(26)与 `test_sparse_indexer_decode_seq_lens.py`(7)在 `5786027c10` 通过;gfx942 kpool 42 cases 通过。
- Sync check:TP4 server `VLLM_GPU_SYNC_CHECK=error` 下两组 bench(8192/256 c16、60000/64 c4)全部请求成功、无 sync 报告。
- 回归风险:`decode_rows_self_filled` 为真时每个 decode return 都必须写满宽(代码中已注释标明),否则上一步的索引残留在 tail 列;测量侧 128k c8 的 +2.1% 在 spread 内;对照为 nightly image 与 nightly+PR 双臂、GPU 半区互换的两轮运行。

## 关联

NVIDIA kpool indexer 与其他模型明确不变。同区域:GLM-5.3-Flash ROCm 相关工作(pack 未提供其他 PR 编号);性能对照组 commit `5786027c10`(rebase 后 `f493d6d757`)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
