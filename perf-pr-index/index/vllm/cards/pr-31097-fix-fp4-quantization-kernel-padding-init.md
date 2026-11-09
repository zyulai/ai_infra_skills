# PR #31097: [FIX] FP4 quantization kernel padding initialization bug

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @danielafrimi |
| 合并 | 2025-12-23 |
| 分类 | Quantization(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/31097 |

## 一句话结论

修复 NVFP4 量化 kernel 中 scale factor padding 区域未正确清零的寻址 bug,合并为单循环遍历含 padding 的全部行列。

## 动机

`nvfp4_quant_kernels.cu` 原实现分两个循环:第一个试图把 scale factor 的 padding 区域置零,第二个处理有效输入。padding 置零循环存在寻址问题,未正确初始化所有必要的 scale factor 内存位置,padding 区可能残留脏值。

## 优化点

统一为单一循环,遍历 `sf_m` x `num_padded_cols`(`num_padded_cols = sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE`,始终对齐到 64)的全部行列:有效区域加载真实数据,越界(行 >= numRows 或列 >= numCols)区域用零填充向量;所有数据都过一遍量化路径以保证 scale factor 被正确计算,但输出只写回有效区域——`out` 张量本身不含 padding,越界写会造成 buffer overrun。另对 `round_up` 补了括号(该改动语义不变,属排版修正)。正文还讨论了与 #30897 引入的 `numKTiles` 计算的关系:在 numCols 为 16 的倍数时,两套 padding 公式恒一致。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_quant_kernels.cu: scaled_fp4_quant kernel 合并 padding 清零与数据量化为单循环;padding 区填零向量,仅对有效区域写 out
```

关键片段(kernel 内统一处理 padding 与有效区):

```cuda
for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
  for (int colIdx = threadIdx.x;
       colIdx < num_padded_cols / CVT_FP4_ELTS_PER_THREAD; ...) {
    // 越界区域用零向量,有效区域加载真实数据
    if (rowIdx >= numRows || elem_idx >= numCols) {
      memset(&in_vec, 0, sizeof(PackedVec));
    } else {
      in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    }
    // out 张量不含 padding,仅有效区域写回
    if (rowIdx < numRows && elem_idx < numCols) out[inOffset] = out_val;
  }
}
```

## 性能声明

Nemotron Nano v3,10k requests,1K/1K input/output,max concurrency 512:TPOT mean 30.54ms→30.26ms,ITL mean 30.51ms→30.23ms,Output throughput 15991.81→16099.50 tok/s;作者结论为"性能基本不变,主要修复正确性问题"。另附 `benchmarks/kernels/bench_nvfp4_quant.py` microbenchmark 结果(正文以截图给出,无文字数字)。

## 验证与回归风险

正文称 "All related tests pass";可见 diff 未含新增测试文件。回归面:所有走 `scaled_fp4_quant`(SM100+ NVFP4)的路径——padding 区域 SF 现在保证清零;非 16 倍数 numCols 下的行为正文未讨论。

## 关联

前序:#30897(numKTiles/padding 计算)。同分类:#21961(FP8 量化)、#65230(NVFP4 Humming 路径)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
