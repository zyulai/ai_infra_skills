# PR #6036: [Kernel] Support Microsoft Runtime Kernel Lib for our Low Precision Computation - BitBLAS

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LeiWang1999 |
| 合并 | 2025-04-22 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动约 14 个文件(另有 5 个文件因超长省略),多为新增文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/6036 |

## 一句话结论

把微软 BitBLAS 混合精度矩阵乘库接入 vLLM,新增 `bitblas`/`gptq_bitblas` 两种量化方法,支持直接加载 GPTQ checkpoint。

## 动机

vLLM 的低比特量化矩阵乘此前主要依赖 Marlin/Machete/AWQ 等 CUDA 实现,精度组合有限。BitBLAS(微软开源)面向 `W_wA_a` 混合精度 BLAS(GPTQ 的 UINT4×FP16、BitDistiller 的 INT2×FP16、BitNet-b1.58 的 INT2×INT8 等),通过硬件感知调优在特定形状上生成更优 kernel,值得作为可选量化后端引入,尤其让现有 GPTQ checkpoint 无需重新量化即可用 BitBLAS kernel 推理。

## 优化点

机制是完整的 QuantizationConfig 插件式接入:新增 `bitblas.py`(BitBLASConfig/BitBLASLinearMethod)与 `gptq_bitblas.py`(GPTQBitBLASConfig/GPTQBitBLASLinearMethod)两个量化方法,注册进 `QUANTIZATION_METHODS` 并加入 config 校验的 optimized 列表。权重按 int8 storage 打包(`pack_factor = 8 // weight_bits`),zeros 支持 "quantized" 模式(gptq_with_bitblas 首选);加载时通过 BitBLAS `Matmul`/`MatmulConfig` 按层形状构造算子,可选 `hardware_aware_finetune(topk=20)` 硬件调优并经 `global_operator_cache` 持久化到本地数据库,避免重复调优。TP 切分侧在 `linear.py` 加 `adjust_bitblas_shard`,按 `bitblas_tile_size` 调整 shard 尺寸/偏移,与 Marlin 机制对齐。checkpoint 兼容:`override_quantization_method` 识别 `checkpoint_format: "bitblas"`(或旧字段 `is_bitblas_format`),GPTQ checkpoint 可直接被 gptq_bitblas 接管。最低支持 capability 70。

## 关键改动

```text
vllm/model_executor/layers/quantization/bitblas.py: 新增 BitBLAS 量化方法(配置 + 线性层 + 权重构造 + apply)
vllm/model_executor/layers/quantization/gptq_bitblas.py: 新增 gptq_bitblas 方法(GPTQ checkpoint 兼容)
vllm/model_executor/layers/linear.py: adjust_bitblas_shard 处理 tile 对齐的 TP 分片
benchmarks/kernels/benchmark_bitblas.py: BLOOM-176B/OPT-65B/LLaMA-70B 形状矩阵的内核基准脚本
docs/source/features/quantization/bitblas.md: 使用文档
```

```python
bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
if enable_tuning:
    bitblas_matmul.hardware_aware_finetune(topk=20)
    global_operator_cache.add(config, bitblas_matmul)
    global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
```

## 性能声明

PR 正文附两张 benchmark 结果图(几个月前评测),文本中无可提取的数字;正文未注明模型/硬件/精度等测试条件的文字描述。仅提供 `benchmark_bitblas.py` 基准脚本(m=1 与 m=8192 两档形状)供复现。

## 验证与回归风险

新增 `tests/models/test_bitblas.py`(bitblas vs gptq checkpoint 的 logprobs top-3 一致性)与 `tests/models/test_gptq_bitblas.py`(GPTQ ckpt 经 bitblas vs 原生 gptq),两者默认 `skipif(True, reason="BitBLAS takes too much time for tuning")` 跳过,且因 BitBLAS 内部锁同步导致的轻微非确定性带 `flaky(reruns=2)`。风险点:首次加载时每层算子需硬件调优,启动耗时显著(测试跳过的原因);依赖 `pip install bitblas>=MINIMUM_BITBLAS_VERSION`;GPTQ 与 bitblas 无 bitwise 一致性,精度仅 top-3 级别对齐;labels 含 v1/multi-modality 等,但实际改动集中在量化层与文档,波及面较窄。

## 关联

同分类(低比特 GEMM 内核):#2497(Marlin INT4 GPTQ)、#20268(Machete zp 支持)、#8248(AWQ Triton 内存优化);均为量化线性层 kernel 的同类替换项。
