# PR #48563: [Perf] [NVIDIA] gemma4: fuse the TP all-reduce with the layer epilogue and FP8 quant on NVLink multicast

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @cognaiger9 |
| 合并 | 2026-11-07 |
| 分类 | Quantization（tier: major） |
| Label | run-ci, jit-kernel |
| Diff | +1031/-28, 6 files（pack diffstat 缺失且 diff 在 40k 截断、注明另有 3 file(s) dropped；计数按 pack 保留的 diff 文本） |
| 链接 | https://github.com/sgl-project/sglang/pull/48563 |

## 一句话结论

把 Gemma-4 dense 层的 TP all-reduce + 归一化 + FP8 量化融合为单个 NVLS 内核，extend TTFT 降 13–22%。

## 动机

PR 之前，Gemma-4 每个 dense 层的 attention 与 MLP 结尾都是同一条链：TP all-reduce → RMSNorm → 残差加（×layer_scalar）→ RMSNorm → per-128 FP8 量化（喂给下一个 block-FP8 GEMM）。TP4、H200、4k-token extend chunk 下这条链是一次 NCCL all-reduce 加三个 epilogue kernel，每处约 301 µs，每层跑两次。已有的 FlashInfer all-reduce fusion 不覆盖 Gemma 的 norm→add→norm 顺序，也不输出 FP8，且面向 decode batch 尺寸设计。

## 优化点

新 JIT 内核 `ar_fusion.cuh`：每个 rank 用 `multimem.ld_reduce` 读入已在 NVSwitch 内求和的行（fp32 累加，reduce-scatter 发生在 switch 里），原地完成两级 RMSNorm、残差更新与 per-128 FP8 量化，再用 `multimem.st` 把 FP8 行与 fp32 scales 多播到所有 rank（all-gather 只有一次 egress 拷贝）。残差流按行分片（row0/rows），只在 owner 上读写、从不交换；barrier 复用 CustomAllReduceV2 pull plane，两次 `multimem.red` 会合。最后一层走 `final` 入口：只做残差更新并以 bf16 多播。attention 与 MLP 两种舍入顺序（round_a）分别匹配 unfused 路径。模型侧为 extend batch 新增 fused layer loop，decode 等其余路径不变；`SGLANG_GEMMA4_AR_FUSION=1` 显式开启，要求 SM90、单节点 TP>1、block-FP8 权重，并按 TP 尺寸设 min/max token 门槛（TP2 默认 2048 起，TP4/8 从 128 起）。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemma4/comm/ar_fusion.cuh: 融合内核（multimem.ld_reduce/st + epilogue + FP8 quant）
python/sglang/kernels/ops/communication/gemma4_ar_fusion.py: JIT wrapper（site / final 两个入口）
python/sglang/srt/layers/communication/gemma4_ar_fusion.py: 对称缓冲区、行分片与 token 门槛
python/sglang/srt/models/gemma4_causal.py: extend 批次的 fused layer loop 与启用条件检查
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py: 接受预量化 (fp8, scales) 输入
```

```python
site:  x = allreduce(partial); a = rmsnorm(x, w_a); r = (a + r) * s
       y = rmsnorm(r, w_b); out, scales = per_token_group_quant_fp8(y, 128)
final: x = allreduce(partial); out = (rmsnorm(x, w_a) + r) * s          (bf16)
```

## 性能声明

Kernel 基准（`bench_gemma4_ar_fusion.py`，H200、hidden 5376、单 site、含拷入 symmetric buffer、CUDA-graph 捕获，µs）：TP4 8192 tokens 569.5→353.9，TP8 8192 561.0→331.4，TP2 8192 472.5→431.1；128 tokens 时 TP4 22.4→14.7（对照为现役 all-reduce + 三个 epilogue kernel）。E2E（`sglang.bench_serving`，`RedHatAI/gemma-4-31B-it-FP8-block`，4×H200 TP4，fa4、bf16 KV、chunked-prefill 8192、random dataset）：median TTFT 每个 cell 降 13–22%（1024/1024 c1：74.13→57.56 ms；8192/128 c64：6888.80→5908.21 ms）；output throughput 增益集中在 prefill-heavy 8192/128 cell（c64 +13.0%：508.06→574.28 tok/s），8192/1024 c64 +5.2%；decode 不走 fused 路径，TPOT 仅在 prefill chunk 与 decode 同步的 cell 变化（8192/128 c64 −9.8%）。

## 验证与回归风险

`test_gemma4_ar_fusion.py` 15 例在 TP2/4/8（H200）通过；GSM8K（1319×4 repeats，thinking off，paired）0.9704→0.9712（Δ+0.0008，95% CI 覆盖 0）；greedy KL drift 4.1e-3/4.8e-3/4.6e-3（4k/16k/48k context，unfused 噪声底 2.2e-3/0.6e-3/1.3e-3），作者归因于 switch 内 fp32 求和与 ring 求和顺序不同。风险：FP8 scale 布局限定 SM90；PP、MoE block、LoRA、per-layer embedding、非 bf16 残差、SP、capture 条件任一不满足即回退常规路径；默认关闭，行为可通过 env 审计。

## 关联

同区域同协议：复用 CustomAllReduceV2 multicast pull plane，与 `all_reduce_residual` 中 Kimi-K3 pull kernels 的信号量协议一致（PR 正文说明）；FlashInfer all-reduce fusion 为未覆盖 Gemma 顺序的既有替代。AMD 侧 all-reduce 工作见同批 #49001。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
