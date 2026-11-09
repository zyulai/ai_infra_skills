# PR #51538: [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lucifer1004 |
| 合并 | 2026-08-15 |
| 分类 | Speculative Decoding (tier: major) |
| Label | bug, speculative-decoding, ready, nvidia, mrv2, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),另有 5 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/51538 |

## 一句话结论

修复 SM120 sparse MLA 上 DSV4 的七个缺陷,打通 plain decode/MTP/DSpark 三种 decode 模式,SWA 宽度调优另带来 13-16% 提速。

## 动机

DeepSeek-V4-Flash-0731 无法可靠运行于 SM120 FlashInfer sparse MLA backend:DSpark 被多个缺陷阻断,且存在一个非 DSpark 特有的 hang(任何 `next_n > 1` 的 MTP server 在 batch 排空后卡死,main 分支既有缺陷,单独立为 issue #51593)。该 backend 因此在三种 decode 模式下都不可端到端使用,本文合并自 #51042 并与维护者协调。

## 优化点

七个修复分两类。功能类:(1) SWA 宽度——非因果 draft batch 的 `decode_swa_indices` 比 `window_size` 宽,但 FlashInfer 路径按 `window_size` reshape 导致 crash,现在 metadata 携带真实 dense width,DSpark 非因果宽度改为 pad 到 64 的倍数(K=5 时 192 = 128 窗口 + 5 draft 条目,而非原来的 256),并依赖 flashinfer#4380 的 dispatch;(2) workspace lane——DSpark target 与 draft CUDA graph 并发持有 workspace 视图,V2 DSpark 为 draft 分配第二条 lane;(3) graph replay/draft KV——修复 `sample_idx_mapping` 零填充导致 capture 写入 slot 0、backbone 输出过早释放、draft KV 写进 null block,以及 draft sampling 改用不相交的 Philox counter 区间;(4) MXFP4 SwiGLU 参数——`FlashInferExperts` 曾对一切 mxfp4 权重注入 GPT-OSS 激活常量(`gemm1_alpha=1.702` 等),DSV4 在 `--moe-backend flashinfer_cutlass` 下走该路径导致生成崩溃,现仅在 quant config 未提供时才注入;(5) SM120 gate 改为按 `(num_q_heads, top_k)` 检查 dispatch 表。稳定性类:(6) MTP 下 padded slot 的 per-token context 长度算出 -1(`0-2+0+1`),被 top-k kernel 按 uint32 读成约 4.29e9,两处 clamp 到 0;(7) `persistent_topk_kernel` 在比较前就把 lengths 转 uint32,异常长度使 CTA 组 leader 永远等待已 early-exit 的 peer(卡在 inter-CTA barrier),现在任何决策前先 clamp 到 `min(stride, max_seq_len)`,`cooperative_topk` 同样 clamp。

## 关键改动

```text
csrc/libtorch_stable/persistent_topk.cuh: 行长在任何分支决策前 clamp 到 min(stride, max_seq_len),消除 hang 与越界读
csrc/libtorch_stable/cooperative_topk.cuh: 长度符号比较后 clamp 到 0,避免负数转 uint32 后输出 0..TopK-1 当作有效索引
vllm/v1/attention/backends/mla/sparse_swa.py: metadata 新增 decode_swa_width 字段,非因果宽度由 get_dspark_swa_index_width 决定(pad 到 64)
vllm/v1/attention/backends/mla/indexer.py: uniform 与 native spec-decode 两条路径的 per-token context 长度 clamp 到 0
vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py: 移除对所有 mxfp4 权重注入 GPT-OSS SwiGLU 常量的逻辑
```

关键 kernel 守卫:

```cuda
const int32_t raw_len = params.lengths[row_idx];
const uint32_t row_bound =
    params.stride < params.max_seq_len ? params.stride : params.max_seq_len;
const uint32_t non_negative_len =
    raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
const uint32_t seq_len =
    non_negative_len < row_bound ? non_negative_len : row_bound;
```

## 性能声明

SWA 宽度从 256 改为 192 后,在 >=8 tokens 时快 13-16%(正文原文:"192 measures 13-16% faster at >=8 tokens",测试条件未在正文中展开)。其余为正确性/稳定性修复:GSM8K(n=1319,DeepSeek-V4-Flash-0731,`--moe-backend flashinfer_cutlass` + `--attention-backend FLASHINFER_MLA_SPARSE_DSV4`,8xRTX PRO 6000 Blackwell):target-only TP4 0.9454;DSpark TP4 0.9477(acceptance 65.5%);DSpark TP4+EP4 0.9462(67.1%);DSpark+P/D(mooncake)0.9492/0.9500。MTP TP4(含 commit 6-7)strict-match 0.9477。隔离 fix 4:去掉该 commit 后 strict-match 0.0000(flexible-extract 0.0220),证明其必要性。

## 验证与回归风险

测试计划列出 8 个测试文件(workspace、dflash prepare inputs、spec decode utils、flashmla sparse、compressor kv cache、flashinfer sparse mla sm120 api、ocp mx moe 等),结果 71 passed / 141 skipped。hang 修复用 drain-shaped reproducer 验证:unfixed 构建 30-180 s 内卡死(4 GPU 100% util / 0% memory util),commit 6 only 13 次全清,commit 7 only(commit 6 revert,负长度仍到达 kernel)3 次全清,证明 kernel 端守卫独立起作用;根因经 cuda-gdb 在真实 hang 上确认。回归风险面:SWA 宽度语义变化触及 ROCm 路径(`amd/rocm.py` 同步改用 `decode_swa_width`)与 FlashMLA sparse metadata;`FlashInferExperts` 的 SwiGLU 参数改动以新增测试钉住 GPT-OSS 行为不受影响;相邻 PR #43970 与 #49139 已确认互补不冲突。

## 关联

同区域:PR #43970(MLA indexer / MTP,同一批文件,互补)、PR #49139(persistent_topk 短行 radix histogram 复用,仍需单独合入);hang 单独立案 issue #51593;DSpark 框架见 pr-46995(DSpark)、pr-47808(DSpark confidence scheduling);合并来源 PR #51042。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
