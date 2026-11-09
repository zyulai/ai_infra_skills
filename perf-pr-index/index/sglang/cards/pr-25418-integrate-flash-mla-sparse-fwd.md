# PR #25418: integrate flash_mla_sparse_fwd

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-06-03 |
| 分类 | Attention / Kernel (tier: major) |
| Label | high priority, quant, deepseek, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用(3 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/25418 |

## 一句话结论

DeepSeek V4 prefill 改用 `flash_mla_sparse_fwd`,kernel 层 1.35x、CUDA wall(无 NCCL)1.10x,并让 32768 chunk prefill 可用。

## 动机

`flash_mla_with_kvcache` 的加载逻辑复杂导致偏慢;切换到 `flash_mla_sparse_fwd` 后 kernel 层观测到 1.35x 提速。另外 chunk prefill 在 chunk size 大于 8192 时失败(32768 会挂),#25502 有详细动机。

## 优化点

在 DeepSeek V4 attention backend 的 extend 路径新增 `_forward_prefill_sparse`:不再调用 `flash_mla_with_kvcache`,而是按请求把 SWA 窗口(始终)与压缩 cache(c4/c128)位置化 gather 到一个扁平 bf16 workspace,再用 per-query rebase 的 indices 直接喂给 `flash_mla_sparse_fwd`(TMA load,绕开复杂加载逻辑)。配套新增:Triton `dequantize_k_cache_paged`(把 v4 分页 FP8 KV cache 直接 dequant 进 workspace 切片,无 `torch.cat`);`SparsePrefillChunkCache` 把 chunk 不变的脚手架(geometry、SWA token ids、combine 输出、workspace)跨层复用,避免每层重建小张量;`combine_topk_swa_indices` Triton kernel 把 topk 索引与 SWA 位置索引合并为一行。触发条件:extend 且 `q.shape[0] > 11673`(_LARGE_INDEXER_QUERY_THRESHOLD)或 env `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1`(默认 False);大 query 同时切到 JIT indexer metadata。长输入也路由到 sparse kernel 以规避 #25484。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_backend.py: _forward_prefill_sparse 主路径与触发条件
python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py: SparsePrefillChunkCache、combine_topk_swa_indices、build_swa_token_ids(新文件)
python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py: 分页 KV cache dequant Triton kernel(新文件)
python/sglang/srt/layers/attention/dsv4/metadata.py: _LARGE_INDEXER_QUERY_THRESHOLD = 11673 与 JIT indexer 路由
python/sglang/srt/environ.py: SGLANG_OPT_FLASHMLA_SPARSE_PREFILL 开关
```

## 性能声明

Prefill 128k,4 x B200,DeepSeek-V4-Flash(DSv4 Flash),`--moe-runner-backend flashinfer_mxfp4`,baseline A vs sparse_prefill B(`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1`),`--profile-by-stage` 单请求:

- CUDA total wall 785.7 ms → 729.4 ms(-7.2%);no-NCCL 701.1 ms → 637.1 ms(-9.1%),Speedup A→B(no-NCCL)1.10x(PR 注明 NCCL kernel wall 波动大,以 no-NCCL 为准)。
- 新增主 kernel `sparse_attn_fwd_kernel` 215 次 146.78 ms,替换掉的 `flash_fwd_splitkv_mla_fp8_sparse_kernel` 215 次 205.89 ms。
- 精度:gsm8k(64-shot,1319 例),DeepSeek-V4-Flash,chunk prefill 8192 得分 0.960(输出吞吐 1342.76 token/s),chunk prefill 32768 得分 0.961(1037.34 token/s)。

## 验证与回归风险

验证手段为 gsm8k 评测与 kernel diff profiling 对比(见上)。风险面:改动集中在 DeepSeek V4 backend 的 extend 路径,decode 与其他模型不受影响;超过 11673 query 的 prefill 会自动切换到 sparse kernel 与 JIT indexer metadata(改变长序列行为以规避 #25484);sparse prefill 默认由 threshold 或 env 开关控制;`SparsePrefillChunkCache` 依赖 copy_ 时置 None 以便 cuda-graph replay 重建。

## 关联

#25502:chunk prefill 修复(动机来源,route 逻辑自其 cherry-pick);#25484:长输入走 `flash_mla_with_kvcache` 的 bug。同区域为 DeepSeek V4 attention/sparse prefill 相关优化。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
