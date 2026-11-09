# PR #36549: MiniMax-M3: allocate the lightning-indexer K cache in fp8 on gfx95

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-24 |
| 分类 | Quantization(major) |
| Label | amd, run-ci, jit-kernel, memory-pool |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/36549 |

## 一句话结论

MiniMax-M3 在 gfx95 上把 lightning-indexer K cache 改为 fp8 分配,index cache 容量减半并换来 +15.6% 的 KV token 容量。

## 动机

MiniMax-M3 的 sparse attention 维护第二条更窄的 KV cache——lightning-indexer K cache,仅用于产生挑选主注意力读取哪些 KV block 的 per-block 分数。它按 `head_num=1, head_dim=sparse_index_dim` 在每个 sparse layer 上分配,ROCm 上一度按模型 dtype(bf16)分配,而主 K/V 跟随 `kv_cache_dtype`;在显存紧张的 MI350 上这块 bf16 index cache 是可减半的开销。

## 优化点

gfx95(MI350/MI355)上把 index cache 以 `fp8_e4m3fn` 分配,由 `SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE` 控制(默认 True),score kernel 以 bf16 q × fp8 k 的 widening 模式运行——与 SM100 `m3_fp8_attn_gemm` 路径已有契约相同,但不需要 fp8 主 KV。Triton qk-norm-rope 内核改为按 index cache 自身 dtype 存储,fp32 只经一次舍入(原先经 bf16 两次);dtype 相同时为 no-op cast。改动是 selection-only:主 K/V 的 `kv_cache_dtype` 不动,仅 `index_dtype` 变化。`get_minimax_sparse_index_dtype()` 作为 index dtype 的单一来源,同时供 pool builder 与 per-token cell size 估算调用,使 sizing 不会与 allocation 漂移(与 DSV4 的 `is_unified_kv_fp8()` 同型)。gfx94x 的 fp8 是 `e4m3fnuz`(max 224),OCP `e4m3fn` 非原生安全,故 MI300 保持 bf16、flag 为 no-op;CUDA 默认配置不变。

## 关键改动

```text
python/sglang/srt/mem_cache/memory_pool.py: 新增 get_minimax_sparse_index_dtype()(fp8_attn_gemm/kv_cache_dtype/model_dtype -> index dtype)
python/sglang/srt/mem_cache/kv_cache_configurator.py: _build_minimax_sparse_kv_pool 改用统一 index dtype 来源
python/sglang/srt/model_executor/pool_configurator.py: _compute_cell_size 的 indexer_dtype_size 用同一 dtype
python/sglang/kernels/ops/attention/minimax_m3_qk_norm_rope.py: idx cache store 直接 cast 到 cache dtype
python/sglang/srt/environ.py: SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE = EnvBool(True)
```

## 性能声明

内存(8×MI350X,MiniMax-M3-MXFP8,TP8,`--mem-fraction-static 0.80`,每行一次 boot):baseline index K cache bf16 43.34 GB、`max_total_num_tokens` 3,189,113、可用显存 41.12 GB;fp8 后 21.67 GB(精确减半)但容量未动、可用显存 62.98 GB(cell-size 估算仍按 bf16);补上 cell-size 计账后 fp8 25.05 GB、`max_total_num_tokens` 3,686,422(**+15.6%**)、可用显存 45.22 GB。压力测试 bs44/80000 input(3,520,000 tokens)干净跑完,峰值 token 使用 0.95,无 OOM 无 retraction。吞吐(同硬件,80k input/600 output,cache hit 90%,warm rep):input throughput +1.88%..+2.90%,output throughput -0.38%..+10.21%(bs32: 934.92→1030.4 tok/s),噪声底 ±0.4%。GSM8K 512 例:两侧均 97.07%。

## 验证与回归风险

未新增单元测试:sizing/allocation 一致性由共享的 `minimax_sparse_index_dtype()` 强制,dtype 选择测试只会复述条件;证据为端到端内存/性能/GSM8K 数据。作者注明这些数字测于 `kvcache.cuh` 改动从本 PR 移除之前与最近 main merge 之前,未重跑。风险面:这是精度变化而非免费融合(fp8 index cache 默认开启),需要 bf16 时设 `SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE=0`;CUDA SM100 fp8 attn-GEMM 模式下 cell-size 修正会略增 `max_total_num_tokens`(未测量);复现时 `--moe-runner-backend aiter` 在 main 上不生效、实际回退 triton(两侧一致,A/B 仍公平)。

## 关联

无具体 PR 编号(机制上对齐 DSV4 的 `is_unified_kv_fp8()` 单一来源模式;SM100 侧既有 `m3_fp8_attn_gemm_enabled` fp8 index cache 路径)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
