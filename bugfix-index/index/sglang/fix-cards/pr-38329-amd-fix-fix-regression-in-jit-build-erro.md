# PR #38329: [AMD][Fix] Fix regression in jit build error with FLUX.2-dev on gfx1250

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#38329](https://github.com/sgl-project/sglang/pull/38329) |
| 作者 | sogalin |
| 合并日期 | 2026-09-09 |
| labels | run-ci, jit-kernel |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | regression |

## 一句话结论
gfx1250 上 FLUX.2-dev 一开始去噪即失败,JIT kernel 编译报错;_blackwell_or_newer 增加 CUDA 构建判断让 ROCm 走回退路径。

## 问题现象
FLUX.2-dev 在 gfx1250(ROCm)上,一旦开始 denoising 就失败,JIT 模块编译报错(日志照抄):

```
Failed to build JIT module sgl_kernel_jit_flux2_gated_resnorm
flux2_gated_resnorm.cuh:57:9: error: invalid output constraint '=&f' in asm
flux2_gated_resnorm.cuh:64:43: error: invalid output constraint '=f' in asm
amd_warp_sync_functions.h:297:62: error: static assertion failed due to requirement
    'sizeof(unsigned int) == 8': The mask must be a 64-bit integer.
3 errors generated when compiling for gfx1250.
```

影响面为 ROCm/gfx1250 环境下 FLUX.2-dev 的推理;CUDA 平台不受影响。

## 根因
正文披露:#37112 新增的 `sgl_kernel_jit_flux2_gated_resnorm` kernel 是 CUDA-only——使用带浮点寄存器操作数约束的 PTX 内联汇编,以及 32-bit mask 的 `__shfl_down_sync`,AMDGPU 两者皆无。该 kernel 之所以被选中,是因为 `_blackwell_or_newer()` 只检查设备 capability;ROCm 下该 capability 由 gfx 编号推导,gfx1250 报告为 (12, 5),能通过 `major >= 10` 的判断,而 HIP 下 `torch.cuda.is_available()` 本来就为 True。与 #34481 属同一类问题,只是发生在其后新增的 kernel 上。

## 修复方案
在 `_blackwell_or_newer()` 中要求 CUDA 构建(增加 `torch.version.cuda is not None`),与 `rope/qwen_qkv_epilogue_jit.py` 对同一 capability 判断已有的 guard 一致。两个入口 `can_defer_flux2_gated_residual()` 与 `can_use_flux2_gated_resnorm()` 都经由该 guard,因此 ROCm 现在对两者都走 `residual_gate_add` + eager norm 回退路径;CUDA 不受影响。

```diff
 def _blackwell_or_newer(device: torch.device) -> bool:
+    # ROCm reports a capability derived from the gfx number, so gfx1250 comes
+    # back as (12, 5) and clears a bare major >= 10 test. The kernel is PTX, so
+    # require a CUDA build like rope/qwen_qkv_epilogue_jit.py does.
     return (
-        torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 10
+        torch.version.cuda is not None
+        and torch.cuda.is_available()
+        and torch.cuda.get_device_capability(device)[0] >= 10
     )
```

## 影响与回归风险
- 影响窗口:自 #37112 引入该 kernel 起至本修复(正文指明 kernel 于 #37112 添加、本问题与 #34481 同类)。
- 回归面:按 diff,仅 `python/sglang/kernels/ops/diffusion/norm/flux2_gated_resnorm_jit.py` 中 `_blackwell_or_newer()` 的平台判断;FLUX.2 gated resnorm 的 kernel 选择在 ROCm 上统一回退到 #37112 之前的非融合路径(正文说明无新数值行为需要验证),CUDA 平台行为不变。

## 关联
- [#37112](https://github.com/sgl-project/sglang/pull/37112)(引入该 CUDA-only kernel)
- [#34481](https://github.com/sgl-project/sglang/pull/34481)(同类问题的先前修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
