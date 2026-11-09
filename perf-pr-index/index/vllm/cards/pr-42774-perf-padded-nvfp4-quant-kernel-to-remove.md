# PR #42774: [Perf] Padded nvfp4 quant kernel to remove additional copy, 2.4%~5.7% e2e performance improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2026-05-18 |
| 分类 | Quantization |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/42774 |

## 一句话结论

nvfp4 量化内核直接输出 CUTLASS 对齐的 padded 激活,消除量化后的 F.pad+contiguous 额外拷贝,kernel 提速 15~40%,e2e +2.4~5.7%。

## 动机

CUTLASS nvfp4 GEMM 要求 K 维对齐;模型 K 非对齐时(如 7152),此前每次前向都要先 `scaled_fp4_quant` 量化激活,再 `torch.nn.functional.pad(...).contiguous()` 补齐到 7168——多出一次全量激活的读写拷贝。该拷贝在 decode 小 batch 下占比显著。

## 优化点

给 `scaled_fp4_quant` 增加 `padded_n` 参数:输出/scale tensor 按 `padded_n` 预分配,量化内核一次写出 padded 结果。CUDA 侧把内核参数拆为 `numCols`(输入有效列,越界按零读入)与新增的 `outputCols`(输出列,按 padded 步长写),padding 区域写零、scale 的 padded 区域同样置零;`cvt_fp16_to_fp4` 与 `cvt_fp16_to_fp4_sf_major` 两条 swizzled/major 路径同步改造。上层 `nvfp4/cutlass.py` 与 `nvfp4/flashinfer.py`(cutlass/cudnn 两个 backend)的 `apply_weights` 删除 `pad_nvfp4_activation_for_cutlass` 调用,改为 `padded_n=x.shape[-1] + weights_padding_bytes * 2` 直接量化到 padded 布局。TRTLLM 8x4 scale layout 暂不支持 padded_n(显式 raise)。

## 关键改动

```text
csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu: 两个量化内核分离 input/output 列,padding 写零
vllm/_custom_ops.py: scaled_fp4_quant/create_fp4_output_tensors 增加 padded_n
vllm/model_executor/kernels/linear/nvfp4/cutlass.py, flashinfer.py: apply_weights 去掉激活 pad 拷贝
tests/kernels/quantization/test_nvfp4_quant.py: 新增 padded 输出正确性测试
```

## 性能声明

Kernel 级(N=8192, K=7152, padded_k=7168,`benchmark_nvfp4_padded_quant.py`,do_bench_cudagraph):batch 1→8192 的 speedup 1.161x~1.428x(如 batch 32:24.42→17.10 us),PR 标题称 15~40% kernel 提升。E2E:`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`,`--quantization modelopt_fp4`,TP=4,`--linear-backend flashinfer_cutlass`,random 数据集 input 64/output 128,并发 128、2000 prompts:mean E2EL 740.57/764.14 → 721.80/721.67 ms,PR 结论为 2.4%~5.7% e2e 提升。要求 compute capability 10.0+;PR 未注明具体 GPU 型号。

## 验证与回归风险

新增 `test_quantize_to_fp4_with_padded_output`:覆盖 swizzled 与非 swizzled 两种 scale 布局,校验 padded 输出形状、有效区域与参考实现一致、padding 区域数据与 scale 均为 0。风险面:仅 cutlass / flashinfer-cutlass / flashinfer-cudnn 路径受益;trtllm 8x4 布局不支持 padded_n;输出 tensor 变大(padded_n//2),下游 `cutlass_scaled_fp4_mm` 直接吃 padded 输入,属接口耦合变更。

## 关联

同区域:NVFP4/modelopt_fp4 量化线(`pad_nvfp4_weight_for_cutlass`、`swizzle_blockscale` 等配套工具);CUTLASS/FlashInfer FP4 GEMM backend PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
