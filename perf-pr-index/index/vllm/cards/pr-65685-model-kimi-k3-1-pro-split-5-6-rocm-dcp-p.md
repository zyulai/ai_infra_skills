# PR #65685: [Model] Kimi K3.1 Pro (split 5/6: ROCm DCP, P/D, indexer top-k)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-11-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, rocm, frontend, speculative-decoding, ready, ci/build, deepseek, kv-connector, nvidia, mrv2, dflash, rust, kimi, k3, scheduler, kv-cache-manager, mooncake, recipe, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65685 |

## 一句话结论

Kimi K3.1 Pro 支持的第 5/6 分片:为 K3.1 DSA 路径(`fp8_ds_mla`、packed hybrid cache)补齐 ROCm DCP、DSA+KDA/mamba packed hybrid cache 的 P/D 传输与 indexer top-k 后端;正文无量化加速数字,NVIDIA 侧以 dummy checkpoint 验证 DCP 路径端到端连通。

## 动机

Kimi K3.1 Pro 的支持工作拆成 6 个分片,splits 1/6–4/6(模型文件、Rust renderer、TP8 KV cache/attention、NVIDIA/AMD 模型更新)已合并;本分片(5/6)承载剩余的 ROCm DCP、P/D 与 indexer top-k 内容(splits 6/6 的 DCP 优化与 #66748 draft 叠在其上)。内容来自拆分前的原始 K3.1 PR,非重复工作:#63859 已为 DeepSeek 的 `ROCM_AITER_MLA_SPARSE` 加 ROCm DCP,本 PR 覆盖的是 K3.1 的 DSA 路径(`fp8_ds_mla`、packed hybrid cache);FlashInfer 0.7.3 MegaMoE kwarg 改名留给 #65974。

## 优化点

三个相互独立的部分:

1. **ROCm Kimi K3.1 DSA under DCP**(gfx950,`fp8_ds_mla`,no sinks):indexer 每个 rank 对自己的 key shard 打分,各 rank 合并本地 top-K 候选(`dcp_indexer_rocm.py`,按 FP32 score 再按 token id 排序);ROCm indexer op 接受 `dcp_rank`、`dcp_world_size`、`cp_interleave`。Sparse MLA 侧,AITER Triton `sparse_mla_fwd` 返回 LSE 供 DCP combine;decode-only step 走 score-only merge(`merge_dcp_scores`)。FlyDSL sparse MLA dispatch(`VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA`)为 opt-in、默认关(kernel 来自 #66532);另含 AITER MLA on packed caches(供 DSA target 的 dense MLA draft 使用)。
2. **P/D for packed hybrid(DSA + KDA/mamba)caches**:Mooncake store——带 tail padding(DSA latent + indexer rows 混在一个 hybrid page)的 manager block 注册为单个 region,而不是在 per-head split 上失败(`group_compact_kernel_blocks`);MoRIIO——packed KDA block 内的 per-layer region offset(`_remote_layer_mr_offset(..., region)`)与 allocation base 上共享一个 MR;NIXL——padded hybrid page 的 region 按 block stride 度量,K3.1 token indexer 保留 packed manager block,小 kernel block 的 DSpark draft 传输整块 packed block,兼容性 hash 加入 `indexer_kv_dtype`。
3. **Indexer top-k**:`SparseIndexerTopk.forward_prefill`(stock indexer 与 GLM-5 Next 使用);FlashInfer `gvr_2` backend,opt-in(`sparse_indexer_topk_backend="flashinfer_gvr2"`)。

## 关键改动

注:正文说明本 PR diff 对 main 共 28 文件、+2.4k −0.1k(约 40% 是测试);pack 的 diff 被截断(注明 28 个文件因篇幅略去),以下仅覆盖已展示部分(全部为测试/CI 文件)。

```text
.buildkite/test_areas/kernels.yaml: MI355 kernel step 加入 dcp_indexer_rocm / FlyDSL sparse MLA / sparse index remap 测试与依赖文件
tests/kernels/attention/test_dcp_indexer_rocm.py(新增): pack_dcp_candidates / select_dcp_candidates 的符号、tie、空 shard 与 CUDA graph replay 测试
tests/kernels/attention/test_rocm_aiter_flydsl_sparse_mla.py(新增): FlyDSL fp8_ds_mla 对照 Triton 参考、>4 GiB cache 64-bit 寻址、回退与 min-tokens gate 测试
tests/kernels/attention/test_rocm_aiter_mla_sink.py: sparse MLA backend 从拒绝 DCP 改为允许
tests/kernels/attention/test_rocm_aiter_mla_sparse_index_remap.py(新增测试): _compact_dcp_indices CSR 压缩、merge_dcp_scores rank-major tie-break 参考对照
tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py: 测试夹具补 decode_context_parallel_size 等配置
```

```python
-def test_sparse_mla_backend_rejects_dcp() -> None:
+def test_sparse_mla_backend_allows_dcp() -> None:
     ...
-    with pytest.raises(ValueError, match="DCP not supported"):
-        RocmPlatform.get_attn_backend_cls(
-            selected_backend=AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
-            attn_selector_config=selector_config,
-        )
+    backend_cls = RocmPlatform.get_attn_backend_cls(
+        selected_backend=AttentionBackendEnum.ROCM_AITER_MLA_SPARSE,
+        attn_selector_config=selector_config,
+    )
+    assert backend_cls == AttentionBackendEnum.ROCM_AITER_MLA_SPARSE.get_path()
```

## 性能声明

- 正文无量化加速数字(pack 的 regex 未检出 perf claims;ROCm 侧的 K3.1 DCP serving 与 GSM8K 验证为 TODO)。
- NVIDIA 连通性验证(GB300 单节点,8-layer dummy checkpoint):`-tp 4 --enable-expert-parallel --decode-context-parallel-size 4 --language-model-only` 下 server 启动,33/33 completions(单发 + 32 并发,200–4.7k token prompts)返回 200;dummy weights,只验证 DCP 路径端到端,不验证准确度。
- P/D:端到端 NIXL / Mooncake / MoRIIO run 为 TODO,目前仅单元测试。

## 验证与回归风险

- 单元测试(GB300,commit 9f3285c559):`tests/v1/kv_connector/unit/moriio`、`test_mooncake_store_layout.py`、`test_mooncake_store_worker.py`、`test_nixl_desc_geometry.py` 全部通过;`tests/kernels/test_top_k_per_row.py`、`test_expand_page_indices.py` 为 TODO。
- ROCm kernel 测试(`test_dcp_indexer_rocm.py`、`test_rocm_aiter_flydsl_sparse_mla.py`、`test_rocm_aiter_mla_sparse_index_remap.py`)在 MI355 kernel step 运行。
- 风险:ROCm(MI355X)K3.1 DCP serving 与 GSM8K 尚未验证;P/D 端到端未跑。共享代码路径(`sparse_attn_indexer.py`、`backends/mla/indexer.py`、ROCm indexer op)只增加可选参数或仅在 ROCm / DCP>1 时走的分支,其他模型默认行为不变;scheduler、model runner、`vllm/v1/core` 无改动。

## 关联

- 拆分关系:1/6 #66030、2/6 #66028、3/6 #66037、4/6 #66532(已合并);6/6 #66027(DCP 优化,叠在本 PR 上);#66748(draft,ROCm DCP global-key indexer prefill 与 AITER all-gather,叠在本 PR 上)。
- #63859:DeepSeek 的 ROCm DCP;#65974:FlashInfer 0.7.3 MegaMoE kwarg rename。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
