# PR #36546: MiniMax-M3: run the sparse prefill main attention through AITER Gluon paged attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-23 |
| 分类 | Attention / Kernel |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但 2 个文件因超长未收录 |
| 链接 | https://github.com/sgl-project/sglang/pull/36546 |

## 一句话结论

MiniMax-M3 稀疏 prefill 主注意力改走 AITER Gluon paged attention,8xMI350X 80k 输入吞吐 +8%。

## 动机

MiniMax-M3 稀疏 prefill 分三步:lightning-index attention 打分、top-k 归约、稀疏主注意力。第三步原是 Triton kernel `flash_prefill_with_gqa_share_sparse`,经 `req_to_token` 逐 128-token block 访问 paged KV pool。本 PR 为其增加 AITER Gluon paged attention(`pa_decode_gluon(..., ps=True)`)第二实现,由 `SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL`(默认 true)控制,不支持的 case 回退 Triton。

## 优化点

主 KV pool 布局不变(仍为 NHD `[max_slots, 1, head_dim]`):新增 Triton gather kernel 把每个请求的 context 收集进持久 SHUFFLE 5D scratch 页(`GLUON_PAGE_SIZE=64`,每个 128-token sparse block 两页),per-query sparse block table 展开为 page table 加 per-query 有效 context 长度;scratch 按 1024 页步长增长、单 buffer 上限 512 MiB,超限抛 `GluonPrefillUnavailableError`,调用方回退原地读 pool 的 Triton 路径。indexer 与 top-k(前两步)不受影响。另带两个共享内核优化:为 `disable_index_value` 层新增 score-only index-attention kernel(跳过 index-value 输出;sparse block 页数不超过 2 时每页只读一个 base slot,替代逐 token 读 `req_to_token`,依赖 allocator 的连续页);Triton 稀疏主注意力在 gfx950 上引入 `SUB_K=64` 的 CDNA KV 子分块,使 QK/PV MFMA 尺寸合适,数值上仅 flash-softmax 重结合、等价。

## 关键改动

```text
python/sglang/srt/layers/attention/minimax_sparse_ops/gluon_prefill.py(新增): gather 到 SHUFFLE 5D scratch、Gluon 调用与静态 gate
python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py: 新增 _index_block_score_only_kernel
python/sglang/kernels/ops/attention/minimax_sparse/prefill/topk_sparse.py: SUB_K 子分块与 gfx95 autotune 配置
python/sglang/srt/environ.py: SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL 开关
```

```python
SPARSE_BLOCK_SIZE = 128
# The Gluon PS kernel takes kv_block_size 16 or 64; 64 halves the block-table width.
GLUON_PAGE_SIZE = 64
PAGES_PER_BLOCK = SPARSE_BLOCK_SIZE // GLUON_PAGE_SIZE
_MAX_SCRATCH_PAGES = (512 * 1024 * 1024) // (_PAGE_ELEMS * 2)
```

## 性能声明

8x MI350X (gfx950)、MiniMax-M3-MXFP8、TP8、80k input / 600 output、cache hit 90%,baseline `upstream/main`,batch size 1/2/4/8/16/32:input throughput +8.08%..+8.49%(BS1 187984→203486 tok/s),output throughput +0.16%..+0.57%;latency BS8 16.27→15.99 s、BS32 33.78→32.65 s。噪声 floor ±0.4%。gsm8k(512 题,temperature 0)97.07%→97.46%。

## 验证与回归风险

静态 gate `can_use_gluon_prefill` 要求:无 sink、`block_size_k==128`、连续单头 NHD pool、bf16/fp16、unit scale 等;scratch 大小逐 forward 检查,OOM 或不支持均回退 Triton。风险:非 gfx950 或其他 dtype/布局走原路径,收益不外溢;host 侧 page layout 依赖 `seq_lens_cpu`;scratch 额外占显存(单 buffer 上限 512 MiB)。

## 关联

baseline 为 upstream/main,正文未列同区域其他 PR 编号。无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
