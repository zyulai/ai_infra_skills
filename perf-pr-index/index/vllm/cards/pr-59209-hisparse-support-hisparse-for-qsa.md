# PR #59209: [Hisparse] Support HiSparse for QSA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lHrHenry233 |
| 合并 | 2026-11-19 |
| 分类 | KV Cache / Memory（tier: major） |
| Label | documentation, ready, ci/build, qwen, kv-connector, kv-cache-manager, hisparse, nixl |
| Diff | pack 标注 +None/-None, None files；pack 末尾说明「12 file(s) dropped from diff」 |
| 链接 | https://github.com/vllm-project/vllm/pull/59209 |

## 一句话结论

把 HiSparse KV offloading 从 DSA/MLA 扩展到 NVIDIA QSA（主 K/V 可下放 host），3P1D disaggregated P/D 吞吐最高 +42.8%、128 roots 下 899 → 1,680 tok/s，代价是低并发单引擎场景 −7.8% 到 −24.0% 吞吐。

## 动机

QSA 模型的主 K/V 目前必须全部驻留 GPU，KV 容量限制了并发与长上下文服务。HiSparse 已为 DSA 提供了 planner 与 index-group runtime（主 K/V offload 到 host，indexer keys、raw-key rings、GDN state 留在 GPU）。本 PR 让 QSA 复用同一套 HiSparse 运行时，使主 K/V 也能下放 host，从而在大负载下维持更大的 running batch。

## 优化点

- **Sparse-role 元数据泛化**：`cache_role`、`topk_buffer_width`、`is_index_group_leader` 移到 `AttentionSpec` 上，把 sparse-role metadata 从 MLA 扩展到 full K/V attention；HiSparse 关闭时 QSA 回退正常 backend。
- **设备缓存 BLNHC 布局**：每个 token 跨本地 heads 的 K/V 构成一条连续传输行，`HiSparseConnector.get_required_kvcache_layout` 由固定 `"BLHNC"` 改为 MLA 用 BLHNC、QSA 用 BLNHC。
- **每 TP rank 私有 host pool**：`host_pool_gib` 按 unique KV heads/replica 计费（padding 与 replicated heads 仍占物理内存）。
- **Fused QSA prepare 写 resident cache**；decode 时 shared resolver 把 logical selections 映射到 resident/hot 物理 KV 行（hot miss 时从 host refill），再走 BF16/FP8 split-K attention：`QSA logical selections -> HiSparse resolver -> physical KV rows -> BF16/FP8 QSA split-K attention`。
- **Prefill & split-K**：prefill 直接读 resident KV 或经 shared staging；split-K 尊重 backing cache 的实际 block stride（含 padding）。
- **MTP & prefix caching**：target 与 draft 层复用 logical selections 但保留独立 physical-index buffer 以支持 CUDA graph replay；prefix 复用限于整 scheduler page。
- **Warmup**：覆盖 physical-index decode；cache-pool block 数变化时 BF16 split-K 不再重新编译（`do_not_specialize` 加入 `num_cache_blocks`）。
- NIXL P/D 支持拆到 #68281（任一 PR 可先合；QSA P-OFF/D-ON 需要两者）；QSA 暂不支持 DCP。

## 关键改动

```text
vllm/config/attention.py: HiSparseConfig 文档改为 "DSA and QSA KV offloading"; AttentionConfig.hisparse_config 注释同步
vllm/config/vllm.py: HiSparse 支持校验放宽为 "DSA models with index_topk and QSA models with indexer_budget"
vllm/distributed/kv_transfer/kv_connector/v1/hisparse/connector.py:
  - return "BLHNC"
  + return "BLHNC" if vllm_config.model_config.use_mla else "BLNHC"
vllm/models/qwen4_exp/nvidia/ops/qsa.py:
  - @triton.jit(do_not_specialize=["num_rows", "num_requests"])
  + @triton.jit(do_not_specialize=["num_rows", "num_requests", "num_cache_blocks"])
  + _qsa_sparse_paged_gqa_splitk_kernel 增加 valid_counts_ptr / stride_valid_counts（physical indices 消费）
vllm/models/qwen4_exp/common/qsa_indexer.py: MLAAttentionSpec 增加 cache_role=SparseCacheRole.INDEXER
vllm/model_executor/layers/attention/mla_attention.py: spec 增加 cache_role=SparseCacheRole.SPARSE（有 topk_buffer_width 时）
vllm/model_executor/layers/attention/sparse_mla_attention.py: staging capacity 抽到 hisparse_staging_block_capacity()
vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py: 增加 HiSparse decode kernel warmup（physical_indices=True）
tests/v1/core/test_kv_cache_planning.py: 新增 test_hisparse_sparse_full_attention_owns_rank_local_complete_kv
tests/models/qwen4_exp/test_qsa_reference.py: 新增 automatic-layout 启动、warmup 覆盖不同 cache pool、hisparse physical indices 消费、prefill 忽略未用 request table 行等用例
```

## 性能声明

（正文注明：GB300 测量先于最新 review 改动，反映测试过的最高负载而非容量上限。）

- **单引擎（single-run，AIPerf 0.13.0，TP2，BF16 main/indexer KV，MTP 与 prefix caching off，32k input / 8k output，32 GiB GPU KV/rank）**：256 clients 时 HiSparse 使 peak running requests 从 64 增至 179，吞吐低 5.3%（3,542 vs 3,354 tok/s）；无 preemption 吞吐峰值：OFF 在 32 clients（2,926 tok/s），ON 在 128 clients（3,414 tok/s，也是整体 ON 峰值）。
- **3P1D disaggregated P/D**（3×TP2 prefill + 1×TP2 decode，仅 D 开 HiSparse；BLNHC、3,328-token pages、FP8 E4M3 main KV、BF16 indexer、FP32 GDN、MTP3；3-start 取中位数）：+38.1% @16 open roots、+42.8% @64 roots；128 roots 达 899 / 1,680 tok/s（off / on，各 1 次 preemption）。明细：8 roots 907/1,158 tok/s（ITL 7.67/6.84 ms）；16 roots fixed replay 782/1,080；32 roots 837/1,324（ITL 36.66/23.90 ms）；64 roots 1,012/1,445（TTFT 19.66/10.42 s）。
- **低并发开销**：单引擎单请求（32k input/512 output）105.66 → 97.40 tok/s（−7.8%）；单引擎 8 请求（60k input/10k output）1,143.21 → 868.84 tok/s（−24.0%）。trace：126 matched steps graph-to-graph 7.52 ms（off）vs 10.11–10.14 ms（on），resolver 0.084 ms/rank/step。实验性 FP8 single-pass kernel 把 256 padded rows 的 attention 从 78 µs 降到 40 µs，但 32 roots 端到端 +0.86% / −3.40%，本 PR 保留 split-K（single-pass kernel 见 #66512）。
- **Host landing（128 roots）**：D GPU cache 首次达 99.75%（31.85/31.93 GiB）时 per-rank host main-KV 用量 21.10/127.97 GiB（后峰值 26.47 GiB）。
- **精度（forced-offload GSM8K，Qwen3.8-Flash-Next-FP8，2×B200 TP2，300 题 20-shot temp=0）**：OFF 297/300，ON 296/300（ON 用 24 GiB host pool，触发 13.4M hot misses / 13.7 GB H2D refill、785 preemptions）；早期 200 题运行：196/200 vs 195/200（FP8 KV+MTP3）、197/200 vs 197/200（BF16）。

## 验证与回归风险

Buildkite #104121 通过：B200 `test_qsa_reference.py` 104 passed；H200 HC/fused prepare/QSA 132 passed、8 skipped（需 Blackwell，已在 B200 通过）。两个 shared-connector draft-mirroring 用例保持 `xfail`（worker fixture 未完成），不提供 MTP passing 覆盖；NIXL 验证在 #68281。回归风险：`cache_role` 进入 `AttentionSpec` 触及所有 KV cache planning 路径（多个既有测试补 `cache_role=SparseCacheRole.SPARSE`）；connector 布局依赖 `use_mla` 分支；QSA 的 P/D 需要 #68281 才完整；单引擎低并发（尤其 8 请求 60k 输入）吞吐回退明显；8/128-root 数据为 preliminary，single-engine ON 行为受 hot rows/host pool 配置影响大。

## 关联

- NIXL P/D：#68281（either PR can merge first；QSA P-OFF/D-ON requires both）。
- Upstream landing policy：#65888；实验性 single-pass FP8 kernel：#66512。
- 早期版本对比见 PR 评论：https://github.com/vllm-project/vllm/pull/59209#issuecomment-6678909131 。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
