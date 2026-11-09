# PR #21787: [Refactor] Remove Duplicate `per_block_cast_to_fp8`, Remove Dependencies of DeepGEMM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-08-01 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 8 个文件(1 个 vllm utils + 1 个 benchmark 脚本 + 6 个测试工具/测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/21787 |

## 一句话结论

把 4 处重复的 FP8 block 量化函数 `per_block_cast_to_fp8` 收敛进 `vllm/utils/deep_gemm.py` 并解除对 DeepGEMM 库的依赖,同时修好损坏的 GEMM benchmark 脚本,为后续 Triton/CUDA 权重量化优化铺路。

## 动机

`per_block_cast_to_fp8` 在仓库里有 4 份拷贝(benchmark 脚本、`tests/kernels/quant_utils.py`、`tests/kernels/moe/modular_kernel_tools/utils.py`、`tests/kernels/moe/test_cutlass_grouped_gemm.py`),签名各不相同;`vllm/utils/deep_gemm.py` 里的版本还依赖运行时 import `deep_gemm.utils.math`。同时 `benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py` 因从 `deep_gemm` 包 import `calc_diff` 等符号而直接 ImportError,无法运行。作者定位这是"用 triton/cuda kernel 优化权重量化"的第一步,需要先有可运行的 benchmark 与统一入口。

## 优化点

纯重构,不改 kernel 本身。机制上做三件事:第一,把规范实现搬进 `vllm/utils/deep_gemm.py`,函数签名扩展为 `per_block_cast_to_fp8(x, block_size=[128,128], use_ue8m0=False)`,`use_ue8m0=True` 时用 `_ceil_to_ue8m0`(把 scale 向上取整到 2 的幂)以匹配 DeepGEMM 在 Blackwell 上的行为,并注明 TODO 后续用 triton/cuda kernel 优化该函数;第二,`_lazy_init` 不再尝试从 `deep_gemm.utils.math` 解析 `_per_block_cast_impl`,`is_blackwell_deep_gemm_used` 的判断改挂在 `_fp8_gemm_nt_impl` 上;第三,benchmark 脚本去掉对 `deep_gemm` 包的直接 import(`calc_diff`/`ceil_div`/`get_col_major_tma_aligned_tensor`/`per_block_cast_to_fp8`/`fp8_gemm_nt` 全部改用 vllm 自有实现),A 的量化也统一走 `per_token_group_quant_fp8`。

## 关键改动

```text
vllm/utils/deep_gemm.py: 收编唯一 per_block_cast_to_fp8(block_size/use_ue8m0 参数),移除 deep_gemm math 依赖
benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py: 修复 ImportError,全部改用 vllm utils
tests/kernels/quant_utils.py: 删除重复实现
tests/kernels/moe/modular_kernel_tools/utils.py、test_cutlass_grouped_gemm.py、test_deepgemm.py、test_block_fp8.py、utils.py: 调用点迁移到统一签名
```

统一后的量化实现(节选):

```python
def per_block_cast_to_fp8(x, block_size=DEFAULT_BLOCK_SIZE, use_ue8m0=False):
    ...
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
```

## 性能声明

PR 正文给出的 benchmark(条件:B200、安装 DeepGEMM,fp8 block dense GEMM,15 个 m/n/k 形状,m∈{64,128,4096})是修复后脚本对三种实现的横向对比,不是本 PR 带来的加速:平均 TFLOPS DeepGEMM 642.50 / vLLM Triton 299.58 / vLLM CUTLASS 639.51;平均加速比 DeepGEMM vs Triton 1.60x faster、DeepGEMM vs CUTLASS 0.74x slower、CUTLASS vs Triton 2.24x faster。小 m 时 Triton 在部分形状快于 DeepGEMM(1.24x/1.13x/1.15x/1.26x/1.14x),m=4096 时全面慢(0.32x-0.47x)。三种实现的平均 diff vs reference 均为 0.000714。正文说明可借此结果指导后续 GEMM 优化方向。

## 验证与回归风险

验证以 benchmark 脚本可运行为准(修复前 ImportError,修复后输出三表);相关测试文件(`test_block_fp8.py`、`test_deepgemm.py` 等)的调用点同步迁移,`test_deepgemm.py` 权重构造显式传 `use_ue8m0=True`。回归风险:签名从无参/`(x, block_size_k, block_size_n)` 统一为 `block_size` 列表,任何未迁移的外部调用点会报错;`use_ue8m0` 路径会把 scale 量化为 2 的幂,数值上与旧默认(直接 `x_amax/448`)不同,仅在 DeepGEMM/Blackwell 相关路径启用;对生产 kernel 行为无改动。

## 关联

正文称这是"用 triton/cuda kernel 优化权重量化"系列的第一步(后续 PR 未在素材中列出);同分类 Quantization 下涉及同一 DeepGEMM 集成面,具体编号素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
