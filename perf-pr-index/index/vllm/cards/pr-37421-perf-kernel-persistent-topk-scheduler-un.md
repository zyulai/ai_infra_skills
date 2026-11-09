# PR #37421: [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-04-08 |
| 分类 | CUDA Graph / Compiler |
| Label | performance, ready, ci/build, v1, deepseek, nvidia |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面);diff 因超长截断,另有 7 个文件未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/37421 |

## 一句话结论

把 DSA indexer 的 TopK 重构为单 persistent kernel 按行动态分派四种算法,长序列微基准最高 2.9x,B200 端到端吞吐 +17%。

## 动机

DSA(DeepSeek-V3.2)decode 每步要对 indexer logits 做每行 top-2048。topK-per-row 依 seq_len 有四种最优算法:短序列(≤8K)用 2048-bin FP16 细直方图一趟完成(平均每 bin 仅 ~4 元素)、中等(8K-64K)用 256-bin 粗直方图 + 多趟 FP32 radix 细化、超长(>64K)超出单 CTA shared memory 需多 CTA 协作 radix select、极短(≤TopK)直接拷贝。前序 #34265 用 CUDAGraph specialization 选 kernel,但 host 侧复杂、需要多个 graph 变体。

## 优化点

改为 persistent scheduler 模式:单一固定 grid 内核(按最坏情况即 large 路径配置 multi-CTA per row group),行循环内按实际 `seq_len` 动态分派 trivial/decode(≤8192,2048-bin)/medium(≤64K,256-bin,shared state 全放 dynamic smem 以免压低 decode 路径 occupancy)/large(>64K,multi-CTA 协作 radix select,经 global memory barrier 的 acquire/release 原语同步,triple-buffered 直方图)四条路径;非 large 行只有组内 CTA 0 工作,其余 CTA 无 barrier 直接跳过;内核自初始化、自清理状态(省去每步一次 `cudaMemsetAsync`,即省一个 CUDAGraph 节点)。grid 形状不变,因此天然 CUDAGraph-safe。BS>32 的 batch 走独立的 FlashInfer FilteredTopK 内核(自 flashinfer PR #2215 改编)。`large_context_topk` 接口替换为 `persistent_topk(logits, lengths, output, workspace, k, max_seq_len)`。

## 关键改动

```text
csrc/persistent_topk.cuh: persistent_topk_kernel + 四路径分派(新增,1321 行)
csrc/ops.h: large_context_topk → persistent_topk 签名
.buildkite/test_areas/kernels.yaml: test_top_k_per_row.py 移入 attention 测试步
```

## 性能声明

微基准(BS∈{1..32},seq_len 4096-163840,单位 µs):短/中序列略有回退(BS1 seq 8192:8.99→9.41,0.95x;BS32 seq 32768:0.83x);长序列大幅领先:BS1 seq 65536 1.49x(28.48→19.09)、96000 2.10x、128000 2.42x、163840 2.58x(54.10→20.97);BS2 seq 163840 2.94x;BS8 seq 163840 2.81x。E2E(NVIDIA B200,`nvidia/DeepSeek-V3.2-NVFP4 -tp 4`,单 prompt input 128000/output 4096):MAIN 输出吞吐 78.64 tok/s、TPOT 12.60 ms → PR 91.87 tok/s、TPOT 10.76 ms,即吞吐提升约 17%(TTFT 持平)。精度:gsm8k eval accuracy 0.954(无前后对照)。

## 验证与回归风险

`tests/kernels/test_top_k_per_row.py` 保留并移入 attention CI 步(与 flashinfer trtllm/cutlass mla 等 attention 内核同组);diff 含 1321 行新 CUDA 头,TopK=2048 固定。风险:中等序列(8K-32K)有 3-17% 回退,由作者接受以换取 CUDAGraph 安全与长序列收益;large 路径依赖 inter-CTA spin-wait 同步,对 grid 配置与 workspace 正确性敏感。

## 关联

取代并关闭:#34265(CUDAGraph-specialization 方案)。同区域(DeepSeek-V3.2 DSA):PR #49792(CuTeDSL fused query kernel)、PR #47327(sparse MLA 短序列 dense MHA 路径)、PR #46076(DSA DCP)。部分 kernel 改编自 flashinfer PR #2215。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
