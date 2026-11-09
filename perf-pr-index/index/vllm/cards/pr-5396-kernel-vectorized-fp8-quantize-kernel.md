# PR #5396: [Kernel] Vectorized FP8 quantize kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @comaniac |
| 合并 | 2024-06-12 |
| 分类 | Quantization (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 2 个文件(CUDA kernel 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/5396 |

## 一句话结论

向量化 FP8 per-tensor 量化 CUDA kernel(4 元素向量存取 + 循环展开 + 乘法替代除法),microbenchmark 提速 1.0x-1.5x,hidden size 越大收益越明显。

## 动机

FP8 量化 kernel(`scaled_fp8_quant_kernel`)原实现为逐元素的 grid-stride 循环,每次读写单个元素、每个元素做一次浮点除法,内存带宽利用率低。受 #5146 启发,作者通过向量化数据传输来更好地吃满 HBM 带宽——量化 kernel 属于典型带宽受限算子,访存效率直接决定耗时。

## 优化点

三项纯 kernel 层优化。第一,inverted scale:`scaled_fp8_conversion` 的参数从 `scale` 换成 `inverted_scale`(kernel 内一次算好 `1.0f / scale`),把逐元素的除法全部换成乘法,省下昂贵的浮点除法指令。第二,循环展开:`#pragma unroll 4` 提升指令级并行(ILP)。第三,向量化访存:定义 `vec4_t<scalar_t>`(`__align__(8)`,4 个输入标量)与 `float8x4_t`(`__align__(4)`,4 个 FP8 输出),把输入/输出指针 reinterpret 成向量类型后按 4 元素为一批搬运 HBM-SRAM 数据;`num_elems >> 2` 个向量迭代处理主体,尾部不足 4 的元素再走标量回退循环,保证非 4 倍数长度正确。kernel 语义不变,仍是 per-tensor scale 的 clamp + FP8 e4m3 转换,同时支持 dynamic(先算 scale)与 static(传入 scale)两种调用。

## 关键改动

```text
csrc/quantization/fp8/common.cu: scaled_fp8_conversion 改用 inverted_scale;scaled_fp8_quant_kernel 向量化(4 元素 vec + unroll 4 + 尾部标量处理)
tests/quantization/test_fp8.py: 新增 test_scaled_fp8_quant(dynamic/static/padding 与非 4 倍数形状)
```

向量化主循环(节选):

```cuda
const float inverted_scale = 1.0f / (*scale);
const vec4_t<scalar_t>* vectorized_in =
    reinterpret_cast<const vec4_t<scalar_t>*>(input);
float8x4_t* vectorized_out = reinterpret_cast<float8x4_t*>(out);
int num_vec_elems = num_elems >> 2;
#pragma unroll 4
for (int i = tid; i < num_vec_elems; i += blockDim.x * gridDim.x) {
  vec4_t<scalar_t> in_vec = vectorized_in[i];
  float8x4_t out_vec;
  out_vec.x = scaled_fp8_conversion(in_vec.x, inverted_scale);
  ...
  vectorized_out[i] = out_vec;
}
```

## 性能声明

PR 正文声明:microbenchmark 显示改进后的 kernel 可达 1.0x-1.5x speedup,尤其当 hidden size 较大时。测试条件:PR 未注明 GPU 型号、hidden size 列表与测量方法(仅称 microbenchmark);无 e2e 数字。

## 验证与回归风险

新增 `tests/quantization/test_fp8.py::test_scaled_fp8_quant`:以与 kernel 完全对齐的参考实现做 allclose,覆盖 fp16/bf16 两种输入 dtype;特意用 `11x11`(size % 4 != 0)覆盖非 4 倍数的尾部路径;并覆盖 dynamic 量化、static 量化与 `batch_dim_padding` 三种调用方式,按 FP8 最低 GPU capability skip。回归风险集中在数值路径:除法改乘法与向量化转换顺序可能带来与旧 kernel 的极小数值差异(测试以反量化后 allclose 验证);该 kernel 是 FP8 per-tensor 量化的公共路径,所有走 `scaled_fp8_quant` 的模型/精度路径都受影响;对齐假设(8/4 字节)依赖输入张量的连续布局。

## 关联

受 #5146 启发(正文明示);其他 FP8 量化优化素材中未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
