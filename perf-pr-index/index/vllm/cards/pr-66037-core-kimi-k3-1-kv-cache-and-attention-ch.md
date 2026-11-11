# PR #66037: [Core][Kimi K3.1] KV cache and attention changes to serve Kimi K3.1 at TP8 (Kimi K3.1 split 3/5)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-11-09 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, speculative-decoding, ready, nvidia, mrv2, dflash, kimi, kv-cache-manager |
| Diff | 素材未提供 diffstat;pack diff 被截断(仅含 2 个测试文件,另 33 files 因体积被省略),可见部分约 +743/-0 |
| 链接 | https://github.com/vllm-project/vllm/pull/66037 |

## 一句话结论

为 Kimi K3.1 在 TP8 服务补齐 KV cache 与 attention 基础:packed hybrid KV cache(BLHNC)、kernel-block slot mapping 与 sparse MLA/DCP 修复,2x4 B300 TP8 dummy 模型可完整服务。

## 动机

这是 Kimi K3.1 支持的 split 3/5(母 PR #65685,拆分评审;split 1/5 #66030 与 split 2/5 已合并,故对 main 的 diff 只含本 split)。K3.1 是 96 头 DSA(DeepSeek-V3.2 式 sparse MLA)模型,TP8 服务前有三个缺口:hybrid 层的 KV page 会被 padding、speculator draft 层需要自己的 slot mapping、以及 ShardQ caller 下 indexer 再按 gathered-row ids 索引会越界(TP8 第一个 mixed prefill/decode step 即 device-side assert)。另外 DSA 层落在 dense backend 上此前会静默跑成 full attention。

## 优化点

- **Packed hybrid KV cache rows(BLHNC)**:模型可用 `get_packed_hybrid_block_size` 自定容纳各层 page 的 block row,使无 page padding;`DS_MLA_STATE_CONTENT_BYTES` 锚定 mamba page 对齐与 packed-cache MLA draft page 大小;hybrid 在 DCP 下保持 layer-compact 布局。
- **Kernel-block slot mapping**:packed token cache 在 runner 侧可拆成 kernel blocks(`AttentionGroup.map_to_kernel_slot_mapping`);speculator 显式 opt-in(`maps_slot_mapping`,DFlash 与 DSpark 开启),draft 层在 target runner 中自成 attention group(DFlash draft 因此多一个 metadata builder);backend 声明 `writes_kv_cache_from_block_table`。
- **Sparse MLA**:DSA 层在 dense backend 上改为直接报错;fp8 query 在量化前 in-place 拼接(32k-token prefill 每层省约 1 ms);ShardQ caller(K3.1)只传本 rank 行的 indexer query/weights,修复越界 assert。
- **K3.1 config**(`KimiK31ForConditionalGeneration` 进 `MODELS_CONFIG_MAP`):DSA 下 kv-cache dtype 默认 per-tensor fp8(FlashInfer sparse MLA 的 SM10x 默认)或 `fp8_ds_mla`;dense draft 配 dense MLA backend。
- **FLASHINFER_MLA_SPARSE**(自 split 1/5 移入本 split):FlashInfer prims_ts(task-scheduled sparse MLA decode)服务 nvfp4_ds_mla 与 DCP 下 per-tensor fp8 cache,需 FlashInfer >0.7.3;trtllm-gen query head padding(head 数 64→128)与 shared multi-CTA KV counters;wide MoE grid 按 SM 数;FA4 MLA prefill 传 `head_size_v`。

## 关键改动

```text
pack diff 仅保留 2 个测试文件(其余 33 files 因体积被省略,以下源码落点来自 PR 正文):
tests/models/kimi_k3/nvidia/specialized_decode/test_specialized_decode_dep.py: DEP attention 对 stock 96-head DSA layer 的逐层等价测试
tests/models/kimi_k3/nvidia/specialized_decode/test_specialized_decode_dsa.py: FlashInfer sparse MLA 的 TP/DCP 核心正确性、精度一致性与 kernel policy 测试
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py: FLASHINFER_MLA_SPARSE 的落点(_prims_ts_row_buckets、FlashInferMLASparseImpl,见测试 import)
```

## 性能声明

无吞吐/延迟 benchmark。验证为功能性:2 节点 TP8(2x4 B300),FlashInfer 0.7.3(main pin),`Kimi-K3.1-8L-dummy`(8 层、DSA 在第 4/8 层、96 heads、MXFP4,`--load-format dummy`):本 PR(be30a1a648)启动 ok——每 GPU 249.9 GiB KV cache、BLHNC 布局、54 PIECEWISE + 64 FULL CUDA graphs;32 并发 completions(78,432 prompt tokens,单请求 200-4,700 tokens)ok、生成 2,048 tokens,与全量源分支(#65685)输出一致;修复 compact-query 之前的分支在同一 warmup 即 device-side assert。PR 明确:该验证不检查 accuracy,真实 checkpoint 不适合本地磁盘。PR 未注明其他测试条件下的量化收益。

## 验证与回归风险

本地 pytest:`test_cudagraph_manager.py` 39 passed、`test_flashinfer_sparse_mla_workspace.py` 12 passed、`test_flashinfer_sparse_mla_sm120_api.py` 5 passed、`test_attn_utils.py` 前后同样失败(本地环境)。CI(#100747/#100906)中 DFlash/DSpark、sparse MLA、FlashInfer sparse MLA、v1 attention job 通过;失败项均在本 PR 未触碰的路径(ROCm MoE kernels、test_dp_supervisor 超时、H200 MIG 5s timing、MI300 ROCM_AITER_MLA)。CI 覆盖缺口:prims_ts 测试需 FlashInfer >0.7.3(CI 中 skip);TP8/两节点服务与真实 checkpoint 不在 CI,由上述手动验证覆盖。风险:DSA-on-dense 从静默降级变为报错;speculator slot-mapping opt-in 影响 DFlash/DSpark metadata builder;ROCm、P/D、indexer top-k 与 DCP 在后续 split(#65685、#66027),本 PR 不含。

## 关联

Kimi K3.1 系列:#65685(母 PR)、#66030(split 1/5)、#66027(DCP split)、split 4/5(ROCm/P-D/indexer top-k,含 NUMA fix)。本批同分类:#59025(FlashInfer backend 注册)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
