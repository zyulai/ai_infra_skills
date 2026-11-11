# PR #63847: [ROCm][Kernel] Support persistent_topk on ROCm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @AndreasKaratzas |
| 合并 | 2026-11-09 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, rocm, speculative-decoding, ready, torch.compile, ci/build, multi-modality, qwen, cpu, kv-connector, nvidia, mistral, dflash, minimax, inkling, nixl, batch-invariance |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数可见 13 个文件,另有 2 个文件因体积被丢弃,diff 末尾截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/63847 |

## 一句话结论

把 vLLM 的 persistent_topk 稀疏 indexer top-k 内核带到 ROCm:MI355X k=2048 下比 `per_row` 快 2.2–2.6×、与 AITER 持平,并修复 decode logits 与 top-k workspace 重叠导致的 row-0 数据损坏。

## 动机

ROCm 上 DSA 稀疏 indexer 的 decode top-k 此前只能用 in-tree `per_row` 或 AITER,请求 `persistent` 后端直接失败——该 op 在 ROCm 构建里被编译为 `STD_TORCH_CHECK(false, "persistent_topk is not supported on ROCm")`。同时接线到 GLM-5.3-Flash 时暴露了一个正确性炸弹:gfx942 上 `rocm_fp8_paged_mqa_logits` 的 decode logits 与 `persistent_topk` 的 row-state workspace 都从共享 workspace offset 0 开始,multi-CTA 路径会覆写 logits(实测 18,964 次调用中 17,346 次返回错误行,全部在第 0 行)。

## 优化点

第一,用 hipcc 构建 `persistent_topk` 及其 FilteredTopK、sampled_topk 各层:新增 wave64 安全的 `TOPK_*` ballot/shuffle 宏(以 32-lane 为一组,wave64 每半个 wavefront 独立运算)、HIP acquire load、clang 兼容的 `exact_topk_rescan` 默认值,并复用 #62936 已合并的候选溢出精确恢复路径,删除了早前 ROCm-only 的 rescan fork。第二,按显卡分派:gfx950(至少 144 KiB LDS)所有 batch 走 `sampled_topk_kernel`(每行一个 workgroup,clustered row 改按精确采样秩截断避免 fallback),gfx942(LDS 64 KiB)保留 persistent multi-CTA 内核,空 batch 提前返回。第三,修 workspace 重叠:`_decode_logits_workspace()` 让 decode logits 排在 top-k workspace 之后申请,gfx942/gfx950/RDN 上所有后端的 logits 现在都落在 workspace 内 1 MiB 偏移处。第四,允许在 ROCm 显式选择 `persistent` 后端(`auto` 仍解析为 `per_row`,默认不变),并把 cooperative/persistent 路由进 `SparseIndexerTopk._row_ends`,修复 `(B,)` seq_lens 且 `next_n>1` 时的 "lengths size mismatch" 报错。

## 关键改动

```text
csrc/topk.cu: 移除 ROCm 编译排除,gfx950/gfx942 分派逻辑
csrc/core/cuda_compat.h: wave64 安全的 SHFL/BALLOT 宏
csrc/sampled_topk.cuh: ROCm 采样路径与 clustered-row 精确秩截断
csrc/persistent_topk.cuh: HIP acquire load 等 hipcc 适配
vllm/model_executor/layers/indexer_topk.py: ROCm 放行 persistent 后端;_row_ends 路由
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: decode logits 移到 top-k workspace 之后
```

```cpp
#ifdef USE_ROCM
  // On gfx950 one workgroup per row beats the multi-CTA kernel
  // at nearly every batch size and length, and needs no workspace reset.
  if (num_rows == 0) return;
  const bool use_sampled = true;
#else
  const bool use_sampled =
      num_rows > 64 &&
      max_seq_len >= vllm::sampled_topk::kMinSampledLength<TopK>;
#endif
  if (use_sampled && max_smem_per_block >= 144 * 1024) { /* sampled_topk */ }
```

## 性能声明

CUDA-graph 内核基准(k=2048,width 163840,us/call,输出逐一校验精确,AITER 0.1.26):clustered 64×128K per_row 311.5 / aiter 133.9 / persistent 60.1;randn 256×128K 102.0/50.3/41.1。全部 36 个形状汇总:persistent 比 `per_row` 快 2.23×、比 AITER 快 1.08×(randn,22/36 领先);clustered logits 下比 `per_row` 快 2.63×、对 AITER 0.98×(9/36 领先);MI355X k=2048 时为 `per_row` 的 2.2–2.6×、与 AITER 相当。端到端 GLM-5.3-Flash(amd/GLM-5.3-Flash-Quark-MXFP4,TP2,MI355X,FULL_AND_PIECEWISE graphs,k=512):本 PR 上 `persistent` vs 默认 8K/1K 851.3 vs 844.3 tok/s(TPOT 15.80 vs 16.12 ms)、32K/1K 386.9 vs 385.8 tok/s;MTP 下 865.3 vs 863.2、405.8 vs 414.0 tok/s。正确性:in-model hook 对 6.94M decode 行/TP rank(gfx950)与 4.48M 行(gfx942 分派)逐行比对 `per_row`,0 行错值;平局时 <0.01% 行索引不同。GSM8K 25-shot:默认 vs persistent 0.9295/0.9287 vs 0.9295/0.9265(各两轮);MI300X 上 per_row vs persistent 0.9196/0.9257 vs 0.9196/0.9166。logits 重叠修复前后对照:3,088 个 logits 被改写 → 0 个。

## 验证与回归风险

`tests/kernels/test_top_k_per_row.py` 改为 ROCm 可跑(含 #62936 的 candidate-overflow cases),新增 ROCm-only `test_persistent_topk_on_rocm_paged_mqa_logits` 锁住 logits 不被覆写;`persistent` 加入 `benchmark_top_k_per_row_decode.py`;CI 的 MI355 DeepSeek V4 kernels step 增加 `-k "persistent or sparse_indexer"` 步骤。gfx950 随机压力测试 2,000 例对 `torch.topk` 0 失败;gfx942 dispatch(MI300X)persistent 组 176 passed,80 skipped ×10 轮。风险:gfx950 全部走 sampled 路径使 multi-CTA 覆盖依赖 MI300 CI;RDNA(wave32)可编译、可放行后端但未经测试;workspace 布局对所有 ROCm 后端都偏移了 1 MiB(正文称未触发 locked workspace);作者提到 2026-10-24 曾见 8/144 次间歇性缓冲损坏,400+ 次复测未复现,归因于共享节点环境。分支 stack 在 #62298 上,合并顺序需协调。

## 关联

前序:#62936(共享的候选溢出精确恢复路径,本 PR 复用其测试);stack 基座 #62298(head b47c6178f1)。同区域:#50470(AITER 稀疏 indexer top-k,互补后端)、#55122(CUDA 上 persistent_topk 平局选择)、#38704(ROCm sparse indexer 分配进 workspace manager)、#41748(改 topk.cu 的 row-state 清理,需与本 PR 互相 rebase)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
