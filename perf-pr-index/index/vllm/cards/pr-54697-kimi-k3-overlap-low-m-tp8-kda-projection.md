# PR #54697: [Kimi-K3] Overlap low-M TP8 KDA projections

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-02 |
| 分类 | Overlap / Scheduler(major) |
| Label | performance, ready, kimi, k3 |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,1 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/54697 |

## 一句话结论

TP8 低 batch decode 时把 Kimi-K3 的 Q/K/V/G 投影与 F_A/beta→F_B 分支跨 stream 重叠,CUDA graph 内投影延迟最高提速 1.391x。

## 动机

Kimi-K3 的 KDA(KimiK3DeltaAttention)在 TP8 下,decode 前向要顺序执行一整条 packed 投影(Q/K/V/G/F_A/beta,权重 [6288, 7168])再接 F_B([1536, 128])。低 M(1-16)时每个 GEMM 都是 skinny 形状,串行执行使 GPU 在小 kernel 间空转;而 QKVG 与 F_A/beta→F_B 两个分支数据无依赖,适合在 CUDA graph replay 中并行。查重说明:与 PR #53525(C=1 intra-kernel PDL handoff,保留投影结构)和 #54151(大 M sequence-parallel AG-GEMM)不重复。

## 优化点

机制分四层:(1) 不拷贝地拆分已有 contiguous packed 权重(QKVG 切片 [0:6144],F_A/beta 为 [6144:6284]),`run_kda_projection_overlap` 用共享多 stream 工具 `maybe_execute_in_parallel` fork/join:主 stream 跑 QKVG,aux stream 跑 F_A/beta→F_B,一对 event 同步;(2) 为 F_A/beta 与 F_B 新增 TP8 专属 CuTeDSL skinny-N/skinny-K GEMM(`kda_skinny_gemm.py`,按 M 分表 split_n,使用 TVM FFI stream 与 PDL),M=1 的 F_B 走 `dsv3_fused_a_gemm`(PDL);(3) QKVG 在 M=1/2 用调优过的 skinny GEMM 配置(KDA_QKVG_CONFIGS),M=3..14 用 autotune 后的 FlashInfer CuTeDSL BF16 GEMM(`mm_bf16`,pdl=True),warmup 阶段 `autotune_kda_qkvg` 提前注册编译;(4) 门控极严:仅 BF16 未量化、TP8、contiguous 生产形状、CUDA graph capture 中且 `VLLM_BATCH_INVARIANT` 关闭时启用,其余走原顺序路径。

## 关键改动

```text
vllm/models/kimi_k3/nvidia/low_latency_gemm.py: run_kda_projection_overlap 分支调度 + 门控 + autotune
vllm/models/kimi_k3/nvidia/kda.py: forward 在 graph capture 且 M<=上限时走 overlap 路径
vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py: 新增 skinny-N/skinny-K CuTeDSL GEMM
vllm/model_executor/warmup/kernel_warmup.py: flashinfer autotune 中注册 QKVG 调优
benchmarks/kernels/benchmark_kimi_k3_kda_projection.py: sequential vs multistream 基准
```

关键片段(分支 fork/join):

```python
projected_qkvg, (g1, beta) = maybe_execute_in_parallel(
    run_qkvg, run_fab_fb,
    events[0], events[1], aux_stream,
)
```

## 性能声明

NVIDIA B300 SXM6 AC,CUDA-graph 投影延迟(µs/projection,4 组权重、300 repeats × 5 trials 取中位数):M=1 16.660→15.719(1.060x)、M=2 20.759→16.467(1.261x)、M=4 23.910→17.194(1.391x,最大)、M=8 22.422→17.305(1.296x)、M=16 21.317→18.887(1.129x);M=3..16 均有 1.06-1.39x 提升。注:正文称门控为 M<=16,但 diff 中最终常量 `KDA_PROJECTION_OVERLAP_MAX_TOKENS = 14`(注释:M=15/16 时原 packed 投影更优,正文 Model Evaluation 一节亦写 "== 16",两处不一致)。正确性:M=1..16 全部输出对 FP32 参考余弦相似度 >0.999;端到端 `moonshotai/Kimi-K3`(MXFP4)TP8 双 GB300 节点:overlap 低 M(max-num-seqs 16)GSM8K strict 96.89%(n=1319)、OCRBench 88.8(888/1000);sequential(max-num-seqs 64)95.53% / 89.1;改动前同配方基线 95.53% / 89.3,判定无精度回归。

## 验证与回归风险

测试:`tests/kernels/test_bf16_skinny_gemm.py` 新增选择器/autotune/CUDA-graph 正确性用例(18 passed,含 TP≠8 不启用、QKVG autotune 后全重叠、capture 后两 stream 输出对拍);`tests/models/kimi_k3/test_kda.py` 57 passed、5 failed(作者说明为本地 stale `_C` 扩展导致的既有失败,报错串 `conv_state must use the SD cache layout` 不在当前上游源码中,本 PR 不改 KDA decode C++ kernel);pre-commit 全过。风险面:仅命中极窄的 TP8+BF16+graph capture 组合,其他配置走原路径;双 stream event 配对与 PDL(griddepcontrol)在非 B300/SM100 硬件上的行为依赖 warmup 注册完整。

## 关联

同区域:PR #53525(KDA 低延迟 GEMM/PDL)、PR #54151(大 M AG-GEMM),正文明确三者互补;共享工具 `maybe_execute_in_parallel` 与 LoRA 双 stream(PR #34933、#42290)同属 multistream overlap 模式。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
