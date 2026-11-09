# PR #36560: MiniMax-M3: wave64 histogram-select decode top-k, and raise kMaxNumBlocks for CUDA graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-23 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36560 |

## 一句话结论

MiniMax-M3 decode top-k 在 ROCm 改用 wave64 histogram-select,输出吞吐 +2.58%~+5.06%,并将 kMaxNumBlocks 提到 16384 解锁 CUDA graph 长上下文。

## 动机

两处瓶颈:其一,ROCm 上 MiniMax-M3 decode block top-k(每个 (head, batch) 行一个 CTA 选 topk 个最高分 KV block)没有针对 wave64 的选择实现;其二,`TopKTrait::kMaxNumBlocks = 4096` 配合 CUDA graph capture——capture 时 `MiniMaxSparseAttnBackend` 以模型**完整** context length(而非 batch 实际 seq lens)约束 `max_seqblock`,即 `ceil(context_len / block_size)`,超过 `4096 × 128 = 524288` tokens 就在 capture 阶段触发 `RuntimeCheck`,top-k kernel 在 graph 下完全不可用。

## 优化点

ROCm 路径新增 `rocm_hist_select`:把 `(score desc, ~block_id asc)` 打包成 u64 全序(NaN 归 -inf,低半 ~id 使并列分时小 block id 胜出),先做一轮粗直方图(≤1024 行用 256 bins,否则 2048),再以 11-bit 轮次 refine 直到候选 ≤128,最后 wave-0 对幸存者做 bitonic sort;高于阈值 bin 的元素立即按 ticket 顺序发射、低于的淘汰、阈值 bin 存活进入下一轮,因此整行数据仍只从 global memory 读取一次。block-id kernel 的 ascending emit 由两个 width-32 shuffle reduction 换成单次 64-lane ballot。`TopKTrait` 模板化为 `TopKTraitImpl<kNumThreads>`,ROCm launcher 在 `max_seqblock > 1024` 时选 1024 线程实例(CUDA 仍只用 512 线程)。`kMaxNumBlocks` 在 ROCm 提到 16384(上限约 2M tokens @ block_size=128),Python 侧 `_MAX_NUM_BLOCKS` 同步并在两个入口断言。

## 关键改动

```text
python/sglang/kernels/jit/csrc/minimax/minimax_decode_topk.cuh: wave64 histogram-select 实现 + trait 模板化 + kMaxNumBlocks 提升
python/sglang/kernels/ops/attention/minimax_decode_topk.py: _MAX_NUM_BLOCKS=16384(HIP)并在两个入口断言
python/sglang/kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py: use_jit_topk 上限改用 _JIT_TOPK_MAX_NUM_BLOCKS
test/registered/kernels/ops/attention/test_minimax_decode_topk.py: HIP 新增 max_ctx=2097152 参数
```

```cpp
#ifdef USE_ROCM
  static constexpr uint32_t kMaxNumBlocks = 16384;  // block topk
#else
  static constexpr uint32_t kMaxNumBlocks = 4096;  // block topk
#endif
static constexpr uint32_t kWaveSize = 64;  // physical wave64
```

## 性能声明

8x MI350X(gfx950),MiniMax-M3-MXFP8,TP8,80k input / 600 output,cache hit rate 90%,CUDA graph 开启,每次启动两轮取第二轮,噪声 ±0.4%,基线 `upstream/main`:

| batch size | 输出吞吐 baseline | 输出吞吐本 PR |
| ---: | ---: | ---: |
| 1 | 55.35 tok/s | 58.15 tok/s |
| 8 | 370.13 tok/s | 383.69 tok/s |
| 32 | 934.92 tok/s | 964.84 tok/s |

输出吞吐 +2.58%~+5.06%,输入吞吐 +0.13%~+0.63%。注意:PR 自述 `--moe-runner-backend aiter` 在 main 上不生效被回退为 triton,但两侧一致,对比仍公平。

## 验证与回归风险

top-k 选择语义不变(同一分数上的另一种归约策略)。gsm8k gate(512 题):97.27% vs baseline 97.07%(`truncated_rate` 1.17% vs 1.56%,`error_rate` 0.00%)。作者明确标注这是**单次 gate**,同系列另一 PR 两次独立运行给出 96.88%/96.29%,run-to-run spread 约 0.6pp,+0.20pp 应读作"在噪声内"。注册测试对 HIP 新增 `max_ctx=2097152` 参数。回归风险:CUDA 路径虽不改选择逻辑,但 `kIters = kMaxNumBlocks / kCTASize` 随常数放大,`num_blocks > 512` regime 每线程寄存器缓存从 8 个 key 升到 32 个(恰在 `static_assert(kIters <= 32)` 上限),寄存器占用上升。

## 关联

无(正文提及"本系列另一 PR"做噪声基线,未给出编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
