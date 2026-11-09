# PR #51309: [ROCm][Perf] Skip redundant sparse index remap on non-indexer layers

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @amd-sriram |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding \| tier: major |
| Label | rocm, ready, verified, mrv1-only |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/51309 |

## 一句话结论

GLM-5.2 在 index_topk_freq=4 下 remap 由 78 降到 21 次/forward，MI355X remap GPU 时间 -73.0%。

## 动机

PR 正文 Summary 指出在 cross-layer index sharing 下只有部分 attention 层会更新共享的 top-k buffer，但旧路径仍在每一层都对该 buffer 做 remap，产生大量冗余。正文给出 Safety 前提：Layer 0 总是拥有 indexer，共享输出在复用前已初始化；非 indexer 层不改变 remap 的输入；默认 `index_topk_freq = 1` 路径不变；speculative decoding 仍每层 remap。

## 优化点

本 PR 只在层拥有 indexer 时运行 remap，GLM-5.2 配 `index_topk_freq = 4` 时每次 forward 的 remap 由 78 次降到 21 次。正文 Update 说明 PR #53781 给 `MLAAttentionSpec` 加了 `is_index_group_leader` 字段（`is_index_group_leader=self.indexer is not None`），使 indexer 层与非 indexer 层落入不同 KV cache group、持有独立 `paged_kv_indices`，因此 remap 需覆盖所有 indexer 层及每组第一个非 indexer 层。实现引入构造期固定的 `index_epoch`（到该层为止的 indexer 写入次数，表示本层有权读哪份 selection）与 metadata 上随 pass 移动的 `remapped_epoch`（该 group 当前承载哪份 selection），二者不匹配才刷新；`forward_mqa` 的 remap 条件为 `self.owns_indexer` 或 `remapped_epoch != index_epoch` 或 `remapped_buf is not topk_indices_buffer`，命中后调 `triton_convert_req_index_to_global_index` 并更新两个标记。

## 关键改动

```text
vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py: 新增 _index_epoch/_INDEX_EPOCHS、owns_indexer/index_epoch，metadata 增加 remapped_buf/remapped_epoch，并把 forward_mqa 的 remap 包进条件判断
tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py: 新增测试（ROCm + AITER 门禁），覆盖 indexer/speculative-decoding gate 与 remap 是否被调用
tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py、test_rocm_aiter_mla_sink.py、tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py: 同步 metadata 与测试夹具
```

```python
        # Indexer layers may rewrite their buffer, so always remap. Non-indexer
        # layers have their own KV cache group: remap once per indexer write.
        if (
            self.owns_indexer
            or attn_metadata.remapped_epoch != self.index_epoch
            or attn_metadata.remapped_buf is not self.topk_indices_buffer
        ):
            # Get topk indices
            assert self.topk_indices_buffer is not None
            topk_indices = fit_kpool_indices_to_aiter(
                self.topk_indices_buffer[:num_actual_toks], attn_metadata.topk_tokens
            )

            triton_convert_req_index_to_global_index(
```

## 性能声明

条件：remap kernel 统计为 MI355X、TP=4、rank 0、same serving workload；model-level e2e 为 Hardware MI355X (gfx950) TP=4，Model `amd/GLM-5.2-MXFP4`、fp8 KV cache，Stack vLLM `69715823` / AITER `4a1cc77` / flydsl `0.3.0`，Server `--async-scheduling --max-num-seqs 256 --max-num-batched-tokens 16384 --block-size 64 --no-enable-prefix-caching`、`index_topk_freq=4`，Env `VLLM_ROCM_USE_AITER=1`、`VLLM_ROCM_USE_AITER_SPARSE_INDEXER=1`、`AITER_USE_FLYDSL_MOE_SORTING=1`、`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`，Perf `vllm bench serve` random dataset `--ignore-eos --request-rate inf`、ISL 1024 与 8192、OSL 1024、concurrency 1-256（16 点），Baseline 为同一镜像仅 revert 本 PR，tok/s 为 4 GPU 聚合输出 tokens/s。原文 remap kernel 表：

| remap kernel | before | after | change |
|---|---:|---:|---:|
| launches | 139,378 | 37,254 | **-73.3%** |
| GPU time | 2,392.1 ms | 645.7 ms | **-73.0%** |
| launches per layer | 1.000 | 0.269 | matches 21/78 |

Serving sweep 共 16 行，Throughput improves on 14 of 16 cases，两处例外均在 concurrency 32（-0.3% 与 -0.0%）；代表行：ISL 1024 conc 1 base 81.3 → PR 81.8 tok/s（**+0.7%**），ISL 1024 conc 256 base 4338.7 → PR 4378.1（**+0.9%**），ISL 8192 conc 256 base 1909.0 → PR 1927.7（**+1.0%**）；geomean：ISL 1024 tput **+0.6%** / TTFT -0.5% / TPOT -0.7%，ISL 8192 tput **+0.6%** / TTFT -1.3% / TPOT -0.6%，all 16 **+0.6%** / -0.9% / -0.6%。Accuracy：gsm8k strict-match 0.9462 → 0.9447、flexible-extract 0.9507 → 0.9469、RULER niah_single_2 @16k 0.870 → 0.884，正文称 Unchanged within run-to-run variation，因跳过的 remap 以 bit-identical 值重写 `paged_kv_indices`。

## 验证与回归风险

Tests：`pytest tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py`，正文称测试覆盖 indexer/speculative-decoding gate 并验证 remap 是否被调用；diff 另有 `test_rocm_aiter_mla_sink.py`、`test_rocm_aiter_mla_sparse_metadata_sync.py`、`tests/v1/attention/test_rocm_aiter_triton_sparse_mla.py`。e2e 用 serving sweep 与 lm_eval gsm8k / RULER 验证精度。回归风险：正文明确默认 `index_topk_freq = 1` 路径不变、speculative decoding 仍每层 remap；改动绑定 ROCm AITER sparse MLA 与 KV cache group 划分（`is_index_group_leader`），依赖 PR #53781，PR 未注明更多受影响路径或精度档位。

## 关联

正文明确：PR #53781 为 `MLAAttentionSpec` 增加 `is_index_group_leader` 字段，使 indexer 与非 indexer 层分入不同 KV cache group，是本 PR remap 条件设计的前提。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
