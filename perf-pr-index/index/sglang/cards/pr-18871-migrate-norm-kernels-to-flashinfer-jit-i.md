# PR #18871: Migrate norm kernels to FlashInfer JIT implementation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Johnsonms |
| 合并 | 2026-03-10 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | sgl-kernel, run-ci |
| Diff | diffstat 缺失(素材未提供统计);pack diff 展示 3 个文件,正文另列 C++ 侧清理(未入 pack diff) |
| 链接 | https://github.com/sgl-project/sglang/pull/18871 |

## 一句话结论

4 个 norm kernel 从 sgl-kernel 静态编译切到 FlashInfer JIT,rmsnorm 对 PyTorch 提速最高 4.05x。

## 动机

关联 issue #17865:把 FlashInfer 外部仓库的 `flashinfer/csrc/norm.cu`(约 5MB)搬进 sgl-kernel 静态编译不可持续。将 rmsnorm、fused_add_rmsnorm、gemma_rmsnorm、gemma_fused_add_rmsnorm 四个 norm kernel 改用 FlashInfer 的 JIT 实现,同时获得性能与可维护性收益,并从 sgl-kernel 的 C++ 构建中移除对应源文件。

## 优化点

`sgl_kernel/elementwise.py` 中四个 norm 入口改为优先调用 `flashinfer.norm.*` 的 JIT 版本;保留原 `torch.ops.sgl_kernel.*` 实现为内部回退(`_rmsnorm_internal` 等),触发回退的条件:MUSA 设备、未安装 flashinfer、dtype 不在 {fp16, bf16}、或 `torch.compiler.is_dynamo_compiling()`——FlashInfer 的 JIT 模块加载路径含 `Path.exists()`/`os.stat()`,在 `torch.compile(fullgraph=True)` 下不可追踪会导致编译失败(flashinfer#2734,修复见 flashinfer#2733,上游合并前先兜底)。函数签名与 import 路径全部不变,属 drop-in 替换。C++ 侧清理:CMakeLists 移除 `norm.cu` 与 `fused_add_rms_norm_kernel.cu` 编译、删除 4 个函数声明与 torch op 注册、删除 wrapper 文件。另新增 `sglang/jit_kernel` 侧的 `bench_norm.py`(SGL JIT kernel vs FlashInfer 对照)与 `test_norm_jit.py`。

## 关键改动

```text
sgl-kernel/python/sgl_kernel/elementwise.py: 4 个 norm 入口切 FlashInfer JIT + 内部回退
sgl-kernel/CMakeLists.txt、include/sgl_kernel_ops.h、csrc/common_extension.cc: 移除静态 norm kernel(正文列出)
python/sglang/jit_kernel/benchmark/bench_norm.py: SGL JIT vs FlashInfer 微基准
python/sglang/jit_kernel/tests/test_norm_jit.py: 对 FlashInfer 参考的正确性测试
```

```python
if (
    input.device.type == "musa"
    or not _has_flashinfer
    or input.dtype not in _FLASHINFER_NORM_SUPPORTED_DTYPES
    or torch.compiler.is_dynamo_compiling()
):
    return _rmsnorm_internal(input, weight, eps, out, enable_pdl)
else:
    return _flashinfer_norm.rmsnorm(input, weight, eps, out, enable_pdl)
```

## 性能声明

对 PyTorch 实现的提速(条件:batch 1-256、hidden 2048-11008 范围内验证;GPU 型号与完整表格在截图与 `bench_norm_jit.py` 中,本卡素材未含数字明细):

- rmsnorm:4.05x
- fused_add_rmsnorm:3.24x
- gemma_rmsnorm:4.43x
- gemma_fused_add_rmsnorm:3.41x

gsm8k(Qwen3-4B,1319 题,8-shot,带/不带 radix cache 两组)结果以截图给出,素材中不可提取。

## 验证与回归风险

sgl-kernel `test_norm.py` 单测:float16 正确性 192/192 通过;bfloat16 在 FlashInfer JIT 侧待支持(32 项失败)。`test_norm_jit.py` 以 flashinfer.norm 为参考,覆盖 fp16/bf16、多个 hidden 与 batch、`specify_out` 两态,断言 rtol/atol 1e-2;测试注释标明约束:JIT rmsnorm 要求 hidden_size 为 256 倍数且 ≤8192,fused_add_rmsnorm 要求 hidden_size % 8 == 0 且 ≤8192。回归风险:所有走 sgl_kernel norm 入口的模型都改经 FlashInfer JIT,受 dtype/hidden 约束的条件依赖回退分支兜底;torch.compile fullgraph 下自动回退静态实现,两实现的数值路径可能存在细微差异;MUSA 与非 CUDA-GPU 路径行为不变。

## 关联

上游:flashinfer#2734/#2733(dynamo 兼容问题)。issue #17865(动机来源)。同区域:#17934(rotary_embedding 的 JIT kernel 迁移,同一 jit_kernel 体系)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
