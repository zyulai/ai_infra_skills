# PR #19233: [Perf] Vectorize static / dynamic INT8 quant kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-06-12 |
| 分类 | Quantization |
| Label | performance, ready |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/19233 |

## 一句话结论

把 `scaled_int8_quant` 四个 kernel 的标量 load/store 换成 VEC_SIZE=16 的对齐向量化读写,w8a8 INT8 端到端吞吐提升约 8%。

## 动机

`scaled_int8_quant` kernel 家族(static/dynamic、含/不含 azp 共 4 个)此前仍是标量 load/store,大量带宽未被利用。issue #18866 指出向量化是下一个主要优化点;PR 基于 #19062、#19109 的增量编译方法展开。

## 优化点

第一,新增 `csrc/quantization/vectorization_utils.cuh` 的 `vectorize_with_alignment<16>` 抽象:按输入指针的实际对齐先做最多 15 个元素的 scalar "prefill",中间主段以 `vec_n_t<InT,16>`(64B 宽度)向量化处理,尾部回落 scalar,从而任意 hidden_size/任意对齐都能吃到向量访存;四个 INT8 量化 kernel 全部改走该抽象,lambda 里只保留 `float_to_int8_rn` 等标量运算。第二,zero-point 重构:引入 `MinMax` 结构体用一次 cub BlockReduce 同时归约 min/max(原来 max/min 两次 Reduce、两次同步),把 min/max 扫描与量化循环合并减少一遍 global memory pass;dynamic kernel 还预计算 `inv_s`(除法变乘法)。第三,block 尺寸从 1024 降为 256(配合 BlockReduce<...,256>),并新增 `benchmarks/kernels/bench_int8_gemm.py` 做 BF16 vs INT8 GEMM 对比 microbenchmark。

## 关键改动

```text
csrc/quantization/vectorization_utils.cuh: 新增 vectorize_with_alignment 通用向量化抽象(供后续 kernel 复用)
csrc/quantization/compressed_tensors/int8_quant_kernels.cu: 四个 INT8 量化 kernel 向量化 + MinMax 单遍归约
benchmarks/kernels/bench_int8_gemm.py: 新增 BF16 vs INT8 GEMM microbenchmark 工具
tests/kernels/quantization/test_int8_quant.py: 增加 hidden_size 1024-1032 向量化边界用例
```

## 性能声明

H100、RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8(`--load-format dummy --input-len 1000 --output-len 100 --max-model-len 32768`):吞吐 before 35.31/34.61 req/s(38822/38027 total tokens/s),after 37.93/38.57 req/s(41685/42403 total tokens/s)。gsm8k 5-shot 前后完全一致(0.7733 flexible-extract / 0.7604 strict-match)。kernel 层面(`bench_int8_gemm.py`,TFLOP/s):int8-tensor-w-tensor-a 在 bs=4096 从 965.33 提到 1111.62,bs=16384 从 953.36 提到 1102.09;int8-channel-w-token-a 在 bs=512 从 790.13 提到 769.39 有回落、大 batch(8192/16384)从 872/877 提到 903/918。

## 验证与回归风险

- 测试:`test_int8_quant.py` 1346 个用例全过(252.81s);新增 HIDDEN_SIZES 1024-1032 覆盖向量化边界(非 16 倍数、跨对齐前缀);gsm8k 精度无回归。
- 风险:vectorize_with_alignment 依赖输入指针与 hidden_size 的运行时处理,边界(极小 hidden_size、未对齐行)由新增用例覆盖;block 1024→256 与 MinMax 归约改变共享内存布局,但仅影响该 4 个 kernel 的调用路径(compressed-tensors INT8 w8a8)。

## 关联

- 前序:#19062、#19109(同系列向量化起点)、issue #18866。
- 同区域后续:#20331(vectorization_utils 进一步优化)、#21476(INT8 per-token-group quant CUDA kernel)、#18844(FP8 quant 提速,同模式)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
