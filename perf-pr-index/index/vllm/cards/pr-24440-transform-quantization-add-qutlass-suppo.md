# PR #24440: [Transform] [Quantization] Add QuTLASS support to vLLM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2025-10-10 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用,且注明 6 个文件因超长未收录 |
| 链接 | https://github.com/vllm-project/vllm/pull/24440 |

## 一句话结论

引入 QuTLASS 库:Blackwell 上 NVFP4/MXFP4 微缩放格式的融合旋转+量化 kernel 与 CUTLASS MXFP4 matmul,配套 fp_quant 量化方法。

## 动机

低精度 4-bit 微缩放(NVFP4/MXFP4)推理中,激活在线量化若与旋转(hadamard 类变换)分离会多出 kernel;vLLM 此前缺少支持在线旋转的量化路径与 Quartet(Quest-like)缩放方案。QuTLASS(https://github.com/iST-DASLab/qutlass,基于 NVIDIA CUTLASS)v0.1.0 提供:在线旋转(fused transform + quantization + scale computation,旋转矩阵运行时加载,支持 16/32/64/128 多种旋转尺寸)、CUTLASS 驱动的 MXFP4 matmul kernel,以及 Quartet 与 Abs-Max 两种量化 scheme。

## 优化点

机制:在 `vllm/_custom_ops.py` 封装 `_qutlass_C` 扩展的 ops——`fusedQuantizeMx`(quest/abs_max 两法,融合旋转、FP4 打包与 e8m0 scale 计算,输出按 128x4 block padding)、`fusedQuantizeNv`(NVFP4 版,e4m3 scale + global_scale)、`matmul_mxf4_bf16_tn`(MXFP4×MXFP4→BF16 matmul)及 ada 变体,均带 fake impl 供 torch.compile;`to_blocked`(qutlass_utils)把 scale 重排为 GEMM 需要的 blocked 布局;NVFP4 GEMM 复用 vLLM 已有的 `cutlass_scaled_fp4_mm`。量化方法注册新 `fp_quant`(FPQuantConfig),测试模型为 ISTA-DASLab 的 Qwen3-0.6B RTN-NVFP4/MXFP4 checkpoint。

## 关键改动

```text
vllm/_custom_ops.py: _qutlass_C ops 封装(fusedQuantizeMx/Nv、matmul_mxf4_bf16_tn 等)
vllm/model_executor/layers/quantization/fp_quant.py: 新 fp_quant 量化方法(6 文件未收录之列)
benchmarks/kernels/bench_mxfp4_qutlass.py / bench_nvfp4_qutlass.py: triton do_bench 微基准
tests/kernels/quantization/test_mxfp4_qutlass.py / test_nvfp4_qutlass.py: RTNE 参考实现逐位对齐测试
tests/quantization/fp_quant.py: e2e 生成测试
```

## 性能声明

微基准与 e2e 结果均以图给出(无文字数字):QuTLASS TFLOP/s 曲线——单层 Qwen3-32B on RTX5090、单层 Llama-3.1-70B on B200,MXFP4 与 NVFP4 各一;e2e `benchmark_latency.py`(Llama-3.3-70B MXFP4-hadamard,B200,WIP)。精度表(Llama,MXFP4,4 项任务平均):FP16 0.882;RTN 0.854 (Recovery 96.8%);RTN+HAD(GS32) 0.863 (97.8%);GPTQ 0.863 (97.9%);GPTQ+HAD(GS32) 0.868 (98.4%)。PR 未注明微基准的具体批大小序列与量化条件以外的测试条件。

## 验证与回归风险

`tests/kernels/quantization/test_mxfp4_qutlass.py`/`test_nvfp4_qutlass.py` 用 torch 参考实现(RTNE fp4 转换、e8m0/e4m3 scale、clip mask)做数值对齐(assert_close 与逐位 equal),覆盖 Llama 7B/13B/33B/70B 全部 4 类层形状、旋转尺寸 32/64/128(MXFP4)与 16/32/64/128(NVFP4)、batch 1/16;要求 CC 10.0/12.0(Blackwell 及以上)。风险面:新量化方法与新 CUDA 扩展依赖(`_qutlass_C`),仅 Blackwell 路径;e2e fp_quant 测试标注 WIP;对非 fp_quant 用户无影响。

## 关联

同区域:NVFP4 量化体系后续演进(如 #53568 的 NVFP4 quant kernel padding 优化);`cutlass_scaled_fp4_mm` 既有 NVFP4 GEMM;gpt-oss/MXFP4 MoE 后端(oracle/mxfp4.py 各后端)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
