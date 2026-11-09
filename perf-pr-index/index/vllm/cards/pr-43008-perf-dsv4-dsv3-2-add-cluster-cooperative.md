# PR #43008: [Perf][DSv4/DSv3.2] Add cluster-cooperative topK kernel for low-latency scenarios

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-06-23 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, ci/build, deepseek, DSv4 |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 截断，省略 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/43008 |

## 一句话结论

为 DeepSeek sparse attention indexer 的 topK 新增 SM90+ cluster 协同 kernel（TMA+DSMEM），低延迟档 microbenchmark 最高约 3 倍、E2E TPOT 改善约 10%。

## 动机

topK v1（persistent_topk，PR #37421）的 persistent scheduler 与多 CTA spin-barrier 协调在低延迟档（bs ≤ 32）过于复杂，且某些配置下 persistent kernel 占满全部 GPU 资源、饿死其他 stream 上的并发工作，还需要 headroom 预分配防止 occupancy 压力下死锁。低延迟场景需要一个释放 SM 及时、不抢占整卡资源的 topK kernel。

## 优化点

- Cluster 协同架构（要求 SM90+，用 TMA 与 DSMEM）：每行一个 cluster，cluster 内多 CTA 各建局部 4096-bin 直方图，`cluster.sync()` 后经 DSMEM distributed SMEM histogram reduction 汇总，再由 `map_shared_rank` 跨块读写完成前缀与输出收集；非 persistent 调度，每行完成即释放 SM。
- 分层路径：sl ≤ 16K 走寄存器直方图（`histogram_4096_topk<12>`，warp-ballot 平局处理 ≤64 个 tie，基本消除 radix refinement 轮次）；`sl/CS ≤ TMA stages × 16K` 走 fused 单 pass（所有 TMA stage 驻留 smem，histogram+scatter 一次完成）；否则 Twopass 双缓冲 TMA streaming（支持任意长序列，验证至 1M）。
- Cluster size 调优：bs ≤ 8 用 CS=8（Blackwell 上 bs ≤ 4 还可用 CS=16），bs 9–32 用 CS=4；warp 归约用硬件 `redux.sync.add` 单条 PTX 指令替代 `__shfl_xor_sync` 蝶形树（正文声明；diff 中 sub-warp 路径仍用 shfl）。
- bs > 32 继承 v1 的 FilteredTopK，但为其加了 `histogram_4096_topk<12, 8>` 快路径（sl ≤ 32K，每线程 32 个 float 的 4096-bin 直方图），大路径不变。

## 关键改动

```text
csrc/libtorch_stable/cooperative_topk.cu/.cuh: 新增 cluster 协同 topK（CS=4/8/16 模板 + TMA/DSMEM 实现）
csrc/libtorch_stable/topk_histogram_4096.cuh: 共享 4096-bin 单 CTA TopK helpers（12-bit bin、tie 处理）
csrc/libtorch_stable/persistent_topk.cuh: FilteredTopK 加 sl<=32K 快路径，移入 filtered_topk namespace
csrc/libtorch_stable/ops.h: 注册 cooperative_topk op（VLLM_ENABLE_COOPERATIVE_TOPK）
.buildkite/test_areas/kernels.yaml: DeepSeek V4 kernel 测试纳入 test_top_k_per_row.py
```

## 性能声明

Microbenchmark（B300，v2/v1 加速比，topK=512）：bs ≤ 32 各 sl（2K-262K）大多 1.3–2.4x，峰值 3.04x（sl 262K, bs=32）；16K–128K 档约 70–80% 提升，262K 约 40%；bs=64/128（FilteredTopK 路径）≈1.0x。topK=1024/2048 形态类似（2048 有个别 0.74x 的回退点，bs=64, sl 262K）。

E2E（B300，DeepSeek-V4-Flash，TP=4，kv fp8，ISL 512K / OSL 2048，C=1，8 prompts）：Mean TPOT 7.90→7.17 ms（约 10%），Output tok/s 58.71→63.00，Total tok/s 14,736.17→15,813.94，P99 TTFT 27,108.63→20,086.67 ms。

C=1 递增 ISL 扫描（3 次平均）：TPOT 改善 +1.5%（64K）至 +7.5%（512K）、+2.1%（1M）；total tok/s +0.6% 至 +5.3%。

Accuracy：GSM8K 双方 0.948；MRCR 2-needle（DeepSeek-V4-Flash, B300, TP=4）各 bucket -0.0005 至 +0.0201。

## 验证与回归风险

- CI：`.buildkite` 将 `tests/kernels/test_top_k_per_row.py` 加入 DeepSeek V4 kernel 测试（含 B200）。
- 风险与限制（PR 明示）：仅 SM90+（TMA/DSMEM），ROCm 上 `TORCH_CHECK(false)`；仅支持 k=512/1024/2048、float32 logits、bs ≤ 32 走协同路径；TMA 要求 `logits.stride(0) % 4 == 0`（用户自定非 4 倍数 `--max-model-len` 会触发 TORCH_CHECK 崩溃，PR 标注该 TODO 已勾选但方案待定）；B200/H200 跨平台 benchmark 与 DSv4-Pro E2E 为未完成 TODO；1M ISL 的 TPOT 增幅略超预期，作者注明需单独研究。

## 关联

- 前序：PR #37421（topK v1 persistent_topk + FilteredTopK，本 PR 的小 batch 路径替代其 persistent 调度并复用其 bs>32 路径）。
- 代码来源：FilteredTopK 改编自 flashinfer PR #2215；`cooperative_topk_body` 部分改编自 sgl-project/sglang PR #23600。
- 同分类：本批 PR #51739、#57057。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
