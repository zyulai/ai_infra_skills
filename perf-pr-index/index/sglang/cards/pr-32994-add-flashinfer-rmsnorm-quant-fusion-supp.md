# PR #32994: Add flashinfer rmsnorm + quant fusion support SM90, SM100, SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DevashishLal-CB |
| 合并 | 2026-08-04 |
| 分类 | Quantization |
| Label | documentation, high priority, performance, quant, sgl-kernel, run-ci, flashinfer, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/32994 |

## 一句话结论

把 RMSNorm 与下游静态 per-tensor FP8 激活量化融合成单个 FlashInfer kernel,H100 e2e 约 +1.5%~2.5%,B200 约 +4%~5%。

## 动机

作者此前尝试 torch.compile 方案做该融合但前景不明,而 flashinfer 新增的 norm+quant kernel(使用了 cluster)比其自写 jit kernel 更快、收益显著,值得直接接入,参考 #10549 与 flashinfer#2243。融合后 FP8 linear 可跳过自己的激活量化步骤。

## 优化点

在 `RMSNorm.forward_cuda` 增加融合路径:当下游 linear 为静态 per-tensor FP8(原生 `Fp8LinearMethod` 且非 block/mxfp8/marlin,或 compressed-tensors W8A8 FP8 static input scheme)时,调用 flashinfer `rmsnorm_quant` / `fused_add_rmsnorm_quant` 直接输出 `(fp8_out, scale, orig_dtype)`;两种 FP8 linear 的 `apply` 检测 tuple 输入后跳过再量化,并用 `orig_dtype` 保持 GEMM 输出 dtype。SM90/SM100/SM120 CUTLASS FP8 GEMM 新增 `ScalarA` 模板分支:`scales_a.numel()==1` 时走 `Sm90ScalarBroadcast` 标量广播,替代原先 per-token 展开 scale 向量的额外拷贝,该项对不开启融合的场景也有小幅提升。新增开关 `--enable-flashinfer-rmsnorm-fp8-quant`,默认关闭;将下游 projection 传入 Llama/Qwen2 的 norm 以便选择融合路径。

## 关键改动

```text
python/sglang/srt/layers/layernorm.py: forward_with_per_tensor_quant_fusion 与 _fp8_static_input_scale 判定
python/sglang/srt/layers/quantization/fp8.py / compressed_tensors_w8a8_fp8.py: apply 消费预量化 (fp8, scale, orig_dtype) 输入
python/sglang/kernels/aot/csrc/gemm/fp8_gemm_kernel.cu: SM90/SM100/SM120 ScalarA dispatch
benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py: 融合微基准(unfused/fused/fused_cute)
```

```python
if quant_linear is not None and not self.cast_x_before_out_mul \
        and _flashinfer_rmsnorm_quant_available:
    scale = _fp8_static_input_scale(quant_linear)
    if scale is not None:
        return self.forward_with_per_tensor_quant_fusion(
            x, scale, residual, post_residual_addition)
```

## 性能声明

kernel 微基准(H100,h4096/h8192,512-16384 tokens):h8192 无 residual、16384 tokens 时 unfused 0.426589 ms → fused 0.183293 ms;h4096 无 residual、16384 tokens 时 0.217082 ms → 0.094088 ms。B200 在线 sweep(concurrency 4-256,isl=osl=1024):output token throughput 均值 +3.12%,mean TTFT +8.89%,mean TPOT +2.78%(TTFT 单点波动 -4.46%~+24.47%)。GSM8K(H100,RedHatAI FP8 模型):Llama-3.1-8B 0.514→0.509,Llama-2-7b 0.255→0.256,Qwen2-7B 0.765→0.765。正文总结:H100 收益 1.5%~2.5%,B200 4%~5%。

## 验证与回归风险

新增 `bench_fused_rmsnorm_fp8_quant.py` 内置 correctness check(fused 与 unfused 在 FP8 精度内一致);`test_fp8_gemm.py` 新增 scalar-A 精度、非法 scale 数拒绝、SM89 拒绝用例;正文称增加 correctness/dispatch/dtype/scalar-broadcast 覆盖。回归面:`fp8_scaled_mm` 放宽了 scales_a 约束(SM89 上 M>1 标量 scale 显式报错),RMSNorm 各后端 forward 签名新增 `quant_linear` 参数;融合仅在静态 per-tensor FP8 且开关开启时触发,block-quant/mxfp8/marlin 与非 FP8 路径不受影响。PR 因 AOT kernel 变更拆为 #33469/#33471 两部分合入。

## 关联

前序 #10549(torch compile 融合尝试)与 flashinfer#2243;后续拆分 PR #33469、#33471。同分类其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
