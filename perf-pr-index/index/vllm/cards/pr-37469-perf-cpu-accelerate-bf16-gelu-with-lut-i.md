# PR #37469: [perf][cpu] Accelerate BF16 GELU with LUT impl on Arm CPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-04-16 |
| 分类 | Other (tier: major) |
| Label | ready, ci/build, cpu |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 8 个文件改动(6 源码/构建 + 1 新增测试 + CI 脚本) |
| 链接 | https://github.com/vllm-project/vllm/pull/37469 |

## 一句话结论

在 Arm CPU 上用 2^16 项查表(LUT)实现 BF16 GELU,GELU 最快加速 8x,量化 Whisper e2e 提速至多 5%。

## 动机

PR 之前 CPU 路径上的 GELU 直接走 PyTorch 原生实现。作者实测发现 PyTorch GELU 可占量化 Whisper 推理 runtime 的约 5%,对一个非 GEMM 的逐元素激活来说开销过大,因此在 Arm CPU 上为其定制快速路径。其他平台行为保持不变。

## 优化点

核心机制是把 BF16 GELU 变成纯查表:BF16 共 2^16 种 bit pattern,LUT 按每个 16-bit 输入值预计算对应的 GELU 输出(经 round_to_nearest_even 转回 BF16),首次调用时用 `std::call_once` + OpenMP 并行初始化,之后每次前向只是 OMP 并行的 `dst[i] = lut[src[i]]` 整数索引,完全省掉浮点计算。实现层面:新增 C++ kernel `activation_lut_bf16` 并通过 TORCH_LIBRARY 注册(仅在 `__aarch64__` 且非 Apple 构建时编译/注册);在 `vllm/model_executor/layers/activation.py` 新增 `GELU` CustomOp,其 `forward_cpu` 在输入为 contiguous BF16 时走 LUT op,否则回退 `forward_native`;并把激活注册表中 `"gelu"` 从 `nn.GELU()` 换成该 CustomOp。CPU 平台的 `check_and_update_config` 在 ARM 上自动追加 `+gelu` 到 custom_ops 以启用该实现。

## 关键改动

```text
csrc/cpu/activation_lut_bf16.cpp: 新增 LUT 激活内核,call_once 预计算 2^16 项 GELU 表,OMP 查表前向
vllm/model_executor/layers/activation.py: 新增 GELU CustomOp,forward_cpu 分派到 LUT op;registry "gelu" 改用它
vllm/platforms/cpu.py: ARM 架构下默认向 custom_ops 追加 "+gelu"
cmake/cpu_extension.cmake / csrc/cpu/torch_bindings.cpp: 构建与 op 注册(aarch64 且非 Apple)
tests/kernels/core/test_cpu_activation.py: 新增 CPU 激活测试
```

核心分派逻辑(节选):

```python
def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
    if self.op and x.dtype == torch.bfloat16 and x.is_contiguous():
        out = torch.empty_like(x)
        self.op(out, x, "gelu")
        return out
    return self.forward_native(x)
```

## 性能声明

PR 正文声明:PyTorch GELU 约占量化 Whisper runtime 的 ~5%;LUT 实现相比现状将 GELU 本身加速 up to 8x;由此量化 Whisper e2e 提速 up to 5%。测试条件:PR 未注明具体硬件型号、线程数与模型规模(仅说明场景为 Arm CPU 上的 quantized Whisper)。

## 验证与回归风险

新增 `tests/kernels/core/test_cpu_activation.py`:对 SiluAndMul/GeluAndMul/GELU/FastGELU/QuickGELU 等激活与 `forward_native` 做 allclose 对比并跑 `opcheck`;GELU+LUT 组合在非 ARM 平台 skip,BF16 以外 dtype 只跳过 opcheck。CI 脚本 `run-cpu-test-arm.sh` 加入该测试文件,作者报告 pytest 通过并将由 CI 覆盖。回归风险:`"gelu"` 注册项从 `nn.GELU()` 换成 CustomOp,影响所有使用 `"gelu"` 激活且跑在 CPU 上的模型路径;LUT 输出为预计算的最近邻舍入值,与原生 GELU 存在表精度差异(测试以 atol/rtol 容差覆盖);非 aarch64 或 Apple 平台不编译该 kernel,行为不变。

## 关联

同区域 CPU 算子优化:PR #48637(INT8 Fused MoE Kernel for Arm CPUs)、PR #27244(Improve cpu fused moe perf)。PR 内未引用其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
