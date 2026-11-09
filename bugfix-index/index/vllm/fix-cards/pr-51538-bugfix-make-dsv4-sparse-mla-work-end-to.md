# PR #51538: [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#51538](https://github.com/vllm-project/vllm/pull/51538) |
| 作者 | lucifer1004 |
| 合并日期 | 2026-08-15 |
| labels | bug, speculative-decoding, ready, nvidia, mrv2, verified |
| 热度 | 👍 2 · 💬 26 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
DeepSeek-V4-Flash-0731 在 SM120 sparse MLA 后端上无法可靠运行,修复 SWA 宽度、workspace、graph 重放、MXFP4 常量、MTP 负长度等 7 个缺陷。

## 问题现象
DeepSeek-V4-Flash-0731 无法通过 SM120 sparse MLA 后端(`--attention-backend FLASHINFER_MLA_SPARSE_DSV4`、`--moe-backend flashinfer_cutlass`)在三种 decode 模式(plain decode、MTP、DSpark)下可靠运行:DSpark 被 commits 1-5 的缺陷阻塞;commits 6-7 修复的挂死不限于 DSpark——batch 排空后任何 MTP(`next_n > 1`)服务都会卡死,属 main 上的既有缺陷(#51593)。复现:drain 型负载(并发 completion 不均匀退出)使未修复构建在 30-180 s 内挂死,4 张 GPU 100% util / 0% memory util。修复后 gsm8k(n=1319):target-only TP4 0.9454,DSpark TP4 0.9477(acceptance 65.5%),DSpark+P/D 0.9492/0.9500,MTP TP4 0.9477;仅移除 fix 4 时 strict-match 降为 0.0000(flexible-extract 0.0220),证实 MXFP4 SwiGLU 常量错误使生成崩坏。

## 根因
正文逐一披露 7 处根因:(1) 非因果 draft batch 分配的 decode_swa_indices 宽于 window_size,FlashInfer DSV4 路径却按 window_size reshape,crash draft;(2) DSpark target 与 draft CUDA 图并发持有 workspace view,每 ubatch 一个 buffer 时一方 resize 会孤儿另一方的存活张量;(3) `sample_idx_mapping` 零填充导致图 capture 执行 padding 行并 scatter 进 request slot 0,capture 的 backbone 输出可能在 replay 读取前被释放,draft KV 可能写进物理块 0(null block);(4) FlashInferExperts 对所有 mxfp4 权重注入 GPT-OSS 激活常量(gemm1_alpha=1.702/beta=1.0/clamp=7.0),DSV4 的 SwiGLU 用错常量;(5) FlashInfer 构建可能暴露 sparse MLA decode API 但缺 DSV4 特化;(6) MTP 下 padding decode slot 的 seq_len==0 使两条 spec-decode 路径算出负 per-token context length(`0 - 2 + 0 + 1 = -1`),top-k kernel 以 uint32 读取变为 ~4.29e9;(7) `persistent_topk_kernel` 在判断 RADIX_THRESHOLD 前就把 lengths 转成 uint32,越界值把行推入 multi-CTA radix 路径,而 host 侧 max_seq_len 的 `cta_in_group != 0` 早退与 per-row 分支不一致,组长永远等在 inter-CTA barrier 上(cuda-gdb 实证:16 个 CTA 仅 1 个存活、spin 在 wait_ge,arrival_counter==1 vs target_val==2),kernel 不退出、engine 挂死。

## 修复方案
按 fix 对应改动:(1) `sparse_swa.py`/`cache_utils.py`/`amd/rocm.py` 用 metadata 携带 `decode_swa_width`,新增 `get_dspark_swa_index_width` 按 64 对齐(K=5 时 192=128+5 draft 补齐,较 256 在 ≥8 token 快 13-16%),kernel 以 `SWA_INDEX_WIDTH` 而非 WINDOW_SIZE 遍历;(2) `gpu_worker.py`/`model_runner.py` 引入第二条 workspace lane,仅 V2 DSpark 启用,speculator 调用包在 `use_workspace_lane` 内;(3) `dflash/speculator.py` 的 `sample_idx_mapping` 改填 -1(惰性行),sliding window 驱逐/null block 的 ctx_slot、q_slot 改写 PAD_SLOT_ID,draft sampling 移到不相交的 Philox counter 区间;(4) `flashinfer_cutlass_moe.py` 删除对一切 mxfp4 注入 GPT-OSS 常量的逻辑(fake_input_scale 仅在 mxfp4+mxfp8 组合保留);(5) `flashinfer_sparse.py` + `utils/flashinfer.py` 新增 `has_flashinfer_sparse_mla_sm120_config`,模型 init 时检查 `(num_q_heads, top_k)` 特化并给出明确报错(需 flashinfer#4380);(6) `indexer.py` 两处对 per-token 长度 clamp 到 0;(7) 两个 topk kernel 先 clamp:
```cpp
const int32_t raw_len = params.lengths[row_idx];
const uint32_t row_bound =
    params.stride < params.max_seq_len ? params.stride : params.max_seq_len;
const uint32_t non_negative_len =
    raw_len > 0 ? static_cast<uint32_t>(raw_len) : 0u;
const uint32_t seq_len =
    non_negative_len < row_bound ? non_negative_len : row_bound;
```
验证:pytest 相关套件 71 passed/141 skipped;commit 6 only 13 次清空、commit 7 only 3 次清空(证明 kernel guard 必要)。注:pack 的 diff 在 40000 字符处截断,以上仅覆盖已展示部分(截断部分未编造)。

## 影响与回归风险
- 影响窗口:正文未给出统一窗口;commits 6-7 属 main 既有缺陷(#51593),MTP 场景自该后端引入即受影响
- 回归面:DSV4 sparse MLA(FLASHINFER_MLA_SPARSE_DSV4)的索引构建与 SWA 宽度、V2 DSpark 的 workspace lane(其他配置 lane 数仍为 1)、dflash speculator 的 CUDA 图重放与 draft KV 写入、flashinfer_cutlass MoE 的 mxfp4 激活常量(新增测试钉住 GPT-OSS 行为不受影响)、两个 topk CUDA kernel 的长度处理、gpt-oss/rocm decode_swa_width 路径;SM120 现在可能在模型 init 即报错(原来在首次 decode 时报不透明 kernel 失败)。

## 关联
- [#51593](https://github.com/vllm-project/vllm/issues/51593)(MTP 挂死 issue 及根因分析)
- [#51042](https://github.com/vllm-project/vllm/pull/51042)(被合并整合的前序 PR)
- [#43970](https://github.com/vllm-project/vllm/pull/43970)(相邻互补,不冲突)
- [#49139](https://github.com/vllm-project/vllm/pull/49139)(persistent_topk.cuh 另一独立 bug,仍需保留)
- [flashinfer-ai/flashinfer#4380](https://github.com/flashinfer-ai/flashinfer/pull/4380)(依赖的 FlashInfer 特化)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
