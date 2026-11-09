# PR #18844: [Perf] Tune `scaled_fp8_quant` by increasing vectorization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-06-03 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 完整可用,共 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/18844 |

## 一句话结论

把 per-tensor/per-token FP8 量化 CUDA kernel 的向量化宽度从 4 提到 16、block 从 1024 降到 256:B200 kernel 提速最高 99%,e2e 输出吞吐 +6.6%~+11.5%。

## 动机

`scaled_fp8_quant` 是 FP8 模型每次激活量化都要经过的显存带宽型 kernel,原实现以 4 元素向量(vec4,8/4 字节对齐)读写、每 block 1024 线程。向量太窄无法吃满显存带宽,block 太大导致每 token 量化(每行一次 BlockReduce)时线程利用率低。改进在 H100 可见、在 B200 上尤为明显。

## 优化点

(1) 向量化宽度 4→16:`thread_max_vec` 与 `scaled_fp8_conversion_vec` 改用 `vec_n_t<scalar_t,16>`/`q8_n_t<fp8_type,16>`(32/16 字节对齐),`num_vec_elems` 移位从 >>2 改为 >>4,内层用 `#pragma unroll` 循环处理 16 个通道;向量化可用条件从 `hidden_size % 4 == 0` 收紧为 `% 16 == 0`,不满足时走标量回退;(2) block 尺寸 1024→256:三个 kernel launch(`static_scaled_fp8_quant`、`dynamic_scaled_fp8_quant`、`dynamic_per_token_scaled_fp8_quant`)统一改 block 256,per-token kernel 的 `cub::BlockReduce<float,1024>` 相应改 256,`segmented_max_reduction` 的 shared cache 降为 256;(3) `max/fabs/min` 全部换成 CUDA 内建 `fmaxf/fabsf/fminf`;(4) `vectorization.cuh` 把 vec4_t/q8x4_t 重构为按 vec_size 参数化的 `vec_n_t`/`q8_n_t` 模板并保留 4 元素别名,layernorm_utils 里的手写 .x/.y/.z/.w 展开改写为 VEC_SIZE 循环(行为等价)。

## 关键改动

```text
csrc/quantization/fp8/common.cuh: thread_max_vec/scaled_fp8_conversion_vec 向量化 16 元素,fmaxf/fabsf 内建
csrc/quantization/fp8/common.cu: 三个量化 kernel launch block 1024→256,BlockReduce<...,256>,向量化对齐检查 %16
csrc/quantization/vectorization.cuh: vec_n_t/q8_n_t 模板化(按 vec_size 对齐),vec4_t/q8x4_t 变为别名
csrc/quantization/fused_kernels/layernorm_utils.cuh: compute_rms/compute_dynamic_per_token_scales/norm_and_quant 改循环写法
```

关键代码:

```cpp
constexpr size_t VEC_SIZE = 16;
using scalarxN_t = vec_n_t<scalar_t, VEC_SIZE>;
int64_t const num_vec_elems = num_elems >> 4;
#pragma unroll
for (int64_t i = tid; i < num_vec_elems; i += step) {
    scalarxN_t in_vec = vectorized_in[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
        absmax_val = fmaxf(absmax_val, fabsf(in_vec.val[j]));
    }
}
```

## 性能声明

B200(`VLLM_ATTENTION_BACKEND=FLASHINFER`,`vllm bench throughput --load-format dummy --input-len 1000 --output-len 100`):静态 per-tensor(RedHatAI/Meta-Llama-3.1-8B-FP8)69.24→73.78 req/s、6924.14→7378.19 output tok/s;动态 per-token(8B-Instruct-FP8-dynamic)60.19→67.09 req/s、6018.77→6709.12 output tok/s;kernel 在 hidden_size=4k、>2k tokens 时提速最高 99%。H100(同命令):37.31→40.92 req/s、3730.78→4091.79 output tok/s;kernel 在 hidden_size=4k、>512 tokens 时提速最高 40%,<256 tokens 有小幅退化(作者认为可接受)。精度:GSM8K 5-shot lm_eval,静态量化 0.4973/0.4973 完全一致;动态量化 flexible-extract 0.7657→0.7650、strict-match 0.7415→0.7407(±stderr 内)。

## 验证与回归风险

测试为 lm_eval GSM8K(静态与动态 FP8 checkpoint)与 e2e throughput 基准,附 kernel 时延图(PR 内截图)。风险面:kernel 属共享路径,所有 FP8(以及 fused rms+quant 路径)用户都经此;向量化对齐要求收紧到 16,hidden_size 非 16 倍数时走标量回退(变慢但正确);<256 tokens 小批量在 H100 有小幅退化;layernorm_utils 重写需保证 residual 更新语义不变。

## 关联

同区域:其他 FP8 量化/activation quant kernel 调优 PR(PR 正文未给编号);与后续 B200 上的 FP8 路径优化(如 #38479 TurboQuant 的 in-kernel FP8 cast)同属显存带宽优化方向。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
