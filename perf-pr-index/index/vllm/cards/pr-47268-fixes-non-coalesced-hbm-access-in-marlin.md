# PR #47268: Fixes non-coalesced HBM access in marlin_int4_fp8_preprocess_kernel_awq

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @flutist |
| 合并 | 2026-07-21 |
| 分类 | Quantization（tier: major） |
| Label | ready |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 1 个文件（约 +10/-8 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/47268 |

## 一句话结论

修复 AWQ int4-fp8 权重预处理 kernel 的非合并 HBM 访问（线程映射行列互换），16384×16384 矩阵上 kernel 提速 2.22 倍。

## 动机

`marlin_int4_fp8_preprocess_kernel_awq` 是 model-load 冷路径上的 AWQ int4-fp8 权重预处理 kernel。原线程映射把 `threadIdx.x` 用在 row 维：相邻 32 线程读的元素按 `size_n/8` 个 int32 word 跨行错开，完全非合并访问，大矩阵上 sector 利用率仅约 12.5%。作者检索 "marlin preprocess" 与该 kernel 的 coalesced-access 修复，未发现重复工作。

## 优化点

纯线程映射重排，无算法改动：把 `threadIdx.x` 换到 column 维（stride 1，完全合并），`blockIdx.x` 直接索引 row。原 `(row=bx*32+tidx, col=by)` 改为 `(row=bx, col=by*32+tidx)`，覆盖同一 `[0,size_k) × [0,size_n/8)` 元素空间且每元素恰好处理一次，逻辑等价。grid 从 `dim3(size_k/32, size_n/8)` 改为 `dim3(size_k, (size_n/8+31)/32)`，并顺带把隐式的 `size_n % 256` 对齐要求放宽为显式 `size_n % 8` 加尾部边界检查（`if (col >= size_n/8) return`），扩大了可处理的矩阵形状范围。

## 关键改动

```text
csrc/libtorch_stable/quantization/marlin/marlin_int4_fp8_preprocess.cu: 重排 kernel 线程映射实现合并访存，放宽 size_n 对齐要求
```

修复后的访存模式：

```cpp
int col = blockIdx.y * 32 + threadIdx.x;
if (col >= size_n / 8) return;
int32_t val = qweight[blockIdx.x * (size_n / 8) + col];
int32_t zero = qzeros[blockIdx.x / group_size * (size_n / 8) + col];
```

## 性能声明

microbenchmark（16384×16384 qweight、group 128、50 次迭代 CUDA event 计时）：before 1291.1 µs/iter → after 581.9 µs/iter，**2.22x** 提速。控制组：未改动的 `without_zp` kernel 路径 557.7 → 557.4 µs/iter（噪声内），证明测量稳定。测试条件仅给出矩阵规模，未注明 GPU 型号。

## 验证与回归风险

正确性：`pytest tests/kernels/quantization/test_marlin_gemm.py::test_marlin_int4_fp8_preprocess_awq` 与 `test_marlin_int4_fp8_preprocess_without_zp` 两用例通过。风险面小：改动限于单一预处理 kernel 的索引计算；对齐要求从 `size_n % 256` 放宽到 `size_n % 8` 意味着更多形状会进入该 kernel（原先可能被拒绝），尾部边界检查覆盖新情况；AWQ zeros 索引同步改为按行分组（`blockIdx.x / group_size`），与新 row 映射一致。

## 关联

同区域为 Marlin/量化预处理相关优化；作者检索未发现针对该 kernel 非合并线程映射的既有 PR。无其他关联编号（素材未列出）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
