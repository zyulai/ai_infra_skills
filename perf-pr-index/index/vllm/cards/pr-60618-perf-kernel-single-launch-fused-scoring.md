# PR #60618: [Perf][Kernel] Single-launch fused scoring+top-k kernel for sparse-indexer prefill, without materializing logits

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-10-31 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, ci/build, deepseek, cpu, DSv4 |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 因超长截断（16 个文件未展开），可见部分为新增 `csrc/fused_mqa_topk/`（candidates/common/constants/deep_gemm_compat/dense_scan 头文件与 `fused_mqa_topk.cu` host wrapper）及 `.buildkite/test_areas/kernels.yaml` |
| 链接 | https://github.com/vllm-project/vllm/pull/60618 |

## 一句话结论

sparse-indexer prefill 换成单 launch 融合 kernel：FP8 打分+top-k 不再落盘 [rows,keys] logits（省 16–128 GiB HBM 往返），1M keys 提速 4.52x，1.04M 上下文 TTFT 降 55%。

## 动机

稠密 prefill 路径把每个 (query row, visible key) 的 fp32 分数写 HBM 再读回做 top-k：一个 16,384 行 chunk 在 128K keys 时产生 8 GiB logits、1M keys 时 64 GiB；受 512 MiB logits 预算限制，稠密路径按 2^27/keys 缩 rows 反复分块——128K 需 76 对调用、1M 需 4,217 对（同一 chunk 上限 127 对）。且该 logits 预算在 memory profiling 时计入峰值内存，与 KV cache 容量直接换手：提高到 8 GiB 预算（128K keys 每 chunk 一次稠密调用）要付出 144,640 个 KV token（−5.6%）。每行只需 k 个分数，往返全是纯开销。

## 优化点

新增 persistent CUDA kernel（`csrc/fused_mqa_topk/`）：每 SM 一个 CTA，从全局计数器按"最重优先"认领 8 行 block，Q/KV/TMEM 流水跨 block 连续。四个阶段：(1) 打分复用 DeepGEMM 的 SM100 FP8 MQA-logits 流水（TMA 加载、UMMA 进 TMEM、warp 专职分工）；利用 relu(s)=(s+|s|)/2，线性半部的加权和化为"权重折叠后的单 query × key"，在 tensor core 上作为额外 UMMA 行完成，math warp 只累加 |s| 项；(2) 采样门限：对 block 的 keys 做跨步采样，直方图相对每行首个采样 tile 的最大值分箱（避免大共同偏移导致门限过松），每行保留数千候选；(3) 候选发射：超门限的 key 由 per-warp 缓冲经两个 drain warp 排入 per-row 候选 slab（6 B/条，每行约 1K–4.5K 条）；(4) 选择：每行一个 warp 在 shared memory 里先采样括号再寄存器直方图分级选 top-k。任何门限异常（候选数不在 [k, 16384]、权重不可安全缩放、候选溢出 scratch）都触发 fp32 精确重扫（稠密打分流水 + streaming top-k merge），输出只在 top-k 边界 tie 上可能与稠密不同。workspace 固定 296 MiB（148-SM B300），与 rows/keys/context 无关。约束：仅 SM100 家族、32 index heads × 128 dims、top-k 512/1024/2048、FP8 indexer cache；可见 keys ≥32K（top-2048 为 64K）才启用。

## 关键改动

```text
csrc/fused_mqa_topk/: 新增持久融合 kernel（打分/门限/候选/选择/精确重扫），host wrapper 全部 scratch 由调用方传入（CUDA-graph 友好）
csrc/fused_mqa_topk/constants.h: 形状常量（32 heads×128 dim、8 行/block、kMaxKeys=1M、每行 16384 候选）
.buildkite/test_areas/kernels.yaml: 加入 csrc/fused_mqa_topk 与新测试
vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py: Python 侧 workspace 镜像（截断部分）
```

```cpp
// 候选记录：fp24 score code（低 16 位 + index word 高 8 位）+ 20 位 key index
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidatesPerRow = 16384, kRetainedPerRow = 4096;
constexpr bool is_supported_topk(int64_t k) {
  return k == 512 || k == 1024 || k == 2048;
}
```

## 性能声明

Kernel 级：B300，真实 GLM-5.x indexer 输入（MRCR 2/8-needle 与 random prompts，层 0/9/19），16,384 行 chunk、top-2048，`do_bench_cudagraph(rep=200)` 中位数。稠密→融合：32K keys 1.05x（2.62→2.43ms）、64K 1.29x、128K 1.49x、256K 1.63x、448K 2.20x、512K 2.40x、944K 4.18x、1M 4.52x（280→62.0ms）；所有 capture 融合 top-k 精确（=fp32 top-k 分数），稠密路径 logits 舍入到 bf16 故边界不精确。E2E：TP4、B300、FP8 KV cache、关闭 prefix caching；GLM-5.2/5.3（21 个 indexer 层全部走 kernel）在 1.04M tokens 处 TTFT 降 711 s（55%）；GLM-5.3-Flash 的 kpool indexer（256K keys@top-512）提速 1.7x（1M keys 稠密为 4.4x）；DeepSeek-V4.1-Flash 仅层 2/8/14 走 kernel，增益较小。KV 容量：融合 296 MiB workspace 使 KV tokens −0.25%（−6,528），稠密 8 GiB 预算则 −5.6%。精度：MRCR 各上下文段与稠密差异均在噪声内（总体 0.561±0.016 vs 0.539±0.016，+1.0σ）。

## 验证与回归风险

测试：新增 `tests/kernels/attention/test_fused_mqa_topk.py` 并接入 kernels CI（含 SM100 job）；真实 capture 验证 fused 与 dense 输出（up to boundary ties）。风险面：仅 SM100（B200/GB200、B300/GB300）；不支持 DeepSeek-V3.2（64 index heads）、MXFP4 indexer cache、DSv4.1 两级候选选择层（层 20/24–36，为后续 PR）；DCP without PCP 不支持（rank 只有 KV shard，保留稠密路径）；GLM-5.3-Flash kpool 不支持 PCP+DCP；端到端仅验证 TP4；>16,384 行 chunk 未验证而排除。

## 关联

致谢与对比：LiteTopK（arXiv:2607.11976，#48726，先驱工作但仅单请求 8192/8128 行、需 2.6 GiB workspace、近似选择 99.998% recall，且存在共享内存越界等正确性问题）。相关：#62011（MXFP4 indexer cache 使 `indexer_sparse_logits` 可用）、#61149（稠密 bf16 舍入讨论）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
