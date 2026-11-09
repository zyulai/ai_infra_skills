# PR #24342: [FP8] Extend per-token-group quantization support to QuantFP8

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tahsintunan |
| 合并 | 2025-09-17 |
| 分类 | Quantization(tier: major) |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 本身可用(5 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/24342 |

## 一句话结论

让 QuantFP8 支持 per-token-group(任意 group size,如 GroupShape(1,128))的 FP8 动态量化,补上 native torch 实现并复用现有 CUDA/Triton group 量化 kernel。

## 动机

对应 issue #24185:QuantFP8 之前用 `assert group_shape in {PER_TOKEN, PER_TENSOR}` 把量化组形状限死在 per-tensor 与 per-token,无法表达 per-token-group(block-wise)量化;需要 group 量化的路径没有统一的 QuantFP8 入口,也没有可与 CUDA 实现对比、可用于 torch.compile 的 native 实现。

## 优化点

`GroupShape` 新增 `is_per_tensor()`/`is_per_token()`/`is_per_group()` 判别方法。`QuantFP8` 构造函数接受任意 group size(`GroupShape(1, N)`),group 量化仅支持 dynamic(static 会 assert 失败),并新增 `column_major_scales` 选项输出列主序 scales。`forward_cuda` 在 group 模式下分流到 `fp8_utils.per_token_group_quant_fp8`(复用既有 CUDA/Triton kernel);新增 `_quantize_group_native` torch 实现:将 hidden 维 pad 到 group size 倍数,按 `(-1, num_groups, group_size)` view 求组内 absmax 得 scale(clamp 到 `_FP8_MIN_SCALING_FACTOR`),量化后裁回原形状,支持 3D/4D 输入与 column-major 转置。`fused_moe.py` 的 `per_token_group_quant_fp8` import 从 `fused_moe.utils` 切到 `fp8_utils`。另将 `benchmarks/kernels/bench_per_token_quant_fp8.py` 扩展为 Torch (Compiled)/CUDA/Triton 三实现、per-tensor/per-token/per-group × 列/行主序 × 多 batch/hidden 组合的对比基准,附 geomean speedup 输出。

## 关键改动

```text
vllm/model_executor/layers/quantization/input_quant_fp8.py: QuantFP8 支持 group 量化(forward_cuda 分流 per_token_group_quant_fp8;新增 _quantize_group_native)
vllm/model_executor/layers/quantization/utils/quant_utils.py: GroupShape 新增 is_per_tensor/is_per_token/is_per_group
vllm/model_executor/layers/fused_moe/fused_moe.py: per_token_group_quant_fp8 改从 fp8_utils 导入
tests/kernels/quantization/test_fp8_quant_group.py: 新增 group 量化正确性测试
benchmarks/kernels/bench_per_token_quant_fp8.py: 三实现对比基准扩展
```

## 性能声明

PR 正文未给出具体性能数字;新增基准脚本用于测 Torch (Compiled) vs CUDA vs Triton 的耗时与 geomean speedup,但正文未附结果。验证以正确性为主:per-tensor、per-token、per-group 三种模式工作正常(CUDA/native 一致性通过)。

## 验证与回归风险

新增 `tests/kernels/quantization/test_fp8_quant_group.py`:小/中/大/不可整除 hidden 维(native-only,如 513/64)、CUDA 与 native scales 一致性(rtol 1e-9)、量化值差异比例 < 0.2%、column-major scales 形状、3D/4D 输入、单 group/全零(scale 被 clamp 到正下限)/极大值(scale > 1)边界;现有 `test_fp8_quant.py` 继续覆盖原有模式。风险:group 量化仅支持 dynamic 的约束、静态量化仍限 per-tensor;fused_moe 的 import 源变更(预期行为等价);CUDA 路径仅覆盖可整除维度,不可整除维度走 native pad。

## 关联

#24185(本 PR 所解决的 issue)。同分类 Quantization 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
