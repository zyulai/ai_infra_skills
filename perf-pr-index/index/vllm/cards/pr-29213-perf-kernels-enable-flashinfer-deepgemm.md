# PR #29213: [Perf][Kernels] Enable FlashInfer DeepGEMM swapAB on SM90 (for W8A8 Linear Op)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @katec846 |
| 合并 | 2026-01-07 |
| 分类 | Quantization |
| Label | documentation, performance, ready, v1, nvidia |
| Diff | diffstat 缺失(pack 未提供 +/− 统计,仅含源码 diff) |
| 链接 | https://github.com/vllm-project/vllm/pull/29213 |

## 一句话结论

SM90 上 W8A8 block-FP8 linear 在 M<32 时改用 FlashInfer DeepGEMM swapAB,小 batch GEMM 提速、conc=4 吞吐 +11%。

## 动机

针对 issue #28427 与 #28316:TRT-LLM 已有 `fp8_gemm_kernel_swapAB` kernel 处理 linear 层的 blockscale FP8 GEMM。vLLM 此前在 SM90 上走 DeepGEMM/CUTLASS,小 M(decode、低并发)场景的 GEMM 不是这些 kernel 的最优形态。但直接把 linear 层全部换成 FlashInfer blockscale FP8 kernel 会在 GSM8K 上出现精度大跌(95%→88%),因此需要一个按 M 分档的安全接入方式。

## 优化点

引入 FlashInfer 的 `fp8_blockscale_gemm_sm90`(swapAB 形态,底层为 TRT-LLM kernel)与配套 FP8 quant kernel,并在 `W8A8BlockFp8LinearOp.apply` 中新增优先分支:当输入/输出为 bf16、weight 为 fp8_e4m3、N%64==0、K%128==0 且 SM90+FlashInfer 可用时走 FlashInfer。kernel 内部按 M 分档——M<32 用 FlashInfer DeepGEMM swapAB,M>=32 用官方 DeepGEMM,FlashInfer 未安装则回退原路径。分档是保精度的关键:M>=32 时 FlashInfer kernel 有明显精度损失,GSM8K 差距达 88% vs 95%。由于 torch.compile 无法捕获输入依赖的 Python 分支,分档用 `torch.cond` 显式注册两条路径,并以 custom op + fake impl 形式接入编译图。整体由新环境变量 `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`(默认关)控制。

## 关键改动

```text
vllm/model_executor/layers/quantization/utils/fp8_utils.py: flashinfer_fp8_blockscale_gemm custom op(torch.cond M<32 分档)与 apply 分支
vllm/utils/flashinfer.py: lazy import、SM90/形状/dtype 支持判定 should_use_flashinfer_for_blockscale_fp8_gemm
vllm/envs.py: 新增 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER 开关(默认 0)
tests/kernels/quantization/test_block_fp8.py: FlashInfer 路径与 native 参考的等价性测试
```

```python
condition = input.shape[0] < 32
return torch.cond(
    condition,
    run_flashinfer_deepgemm_swapAB,
    run_deepgemm,
    (input, weight, weight_scale),
)
```

## 性能声明

- Serving:DeepSeek-V3.1,8xH200 TP8,conc=4,random 数据集 input/output 各 1000 token:输出吞吐 314.70→350.57 tok/s(**1.11x more otps**),TPOT 12.52→11.22 ms(基线带 `VLLM_USE_DEEP_GEMM=1` 等,PR 侧加 `VLLM_USE_FLASHINFER_FP8_LINEAR=1`)。
- OTPS/user speedup 扫描:DS-V3.1 TP8 conc=1/2/4/8/16 为 1.13/1.21/1.11/1.08/1.05,conc>=32 约 1.00;Qwen3-235B-A22B-FP8 TP4 为 1.10/1.09/1.06/1.05/1.01,TEP4 为 1.15/1.14/1.06/1.07/1.01。
- Nsys(DS-V3.1 TP8 conc=4):attention layer 89us→75us(**1.18x faster**),gen step 12.46ms→11.28ms(**1.10x faster**)。

## 验证与回归风险

新增 `test_w8a8_block_fp8_flashinfer_matmul`,与 `native_w8a8_block_matmul` 比对 rel_diff<0.001(需 SM90+FlashInfer,fp8_fnuz 平台跳过)。精度做了 bs=16/32/auto × flexible/strict 的完整矩阵:Qwen3-235B-A22B-FP8 与 DS-V3.1 各档与 Main 差异在 ±1% 左右,无系统性下降。风险:默认关闭,需显式开 env;`torch.cond` 双分支同时进编译图,可能增加显存/编译开销——PR 精度日志中 PR 侧 init engine 耗时 647.95s vs Main 145.81s(疑似一次性编译开销,PR 未解释);若未来放开 M>=32 用 FlashInfer 需重验精度。

## 关联

动机来自 issue #28427、#28316与 FlashInfer PR flashinfer#2131。同批 swap AB 方向:#36559(fused_moe_kernel 的 swap AB);PR 正文亦引用 #20396 作同类先例(经 #36559 正文)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
