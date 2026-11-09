# PR #41197: [ROCm][benchmark][perf] fix moe tuning script for RDNA

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#41197](https://github.com/vllm-project/vllm/pull/41197) |
| 作者 | amd-xavierwang |
| 合并日期 | 2026-10-20 |
| labels | performance, rocm, verified |
| 热度 | 👍 5 · 💬 9 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
RDNA 上 MoE 调参脚本慢、崩溃挂死 ray worker;剔除 CDNA 专属参数、按 VGPR 剪枝并修复 Triton 3.6+ 缓存清理。

## 问题现象
`fused_moe_triton` kernel 的调参对性能影响巨大——最优 .json 可让 GLM4.7-Flash 在 gfx1100 上提速 +15%——但现有 benchmark 脚本非常慢(8 GPU 下超过 4 小时),原因是搜索空间过大,且伴随崩溃,会让 ray worker 无限期挂起。针对 RDNA 具体表现为:某些 Triton 版本下 gfx12 使用 `matrix_instr_nonkdim=32` 会让 `tl.dot` 硬失败报 "no matching matrix core intrinsic"(tensorcore/WMMA 恒为 16*16*16);部分坏配置还会因 cudagraph 撞上非法内存引发底层 kernel 崩溃。

## 根因
正文披露三点:
1. `matrix_instr_nonkdim` 与 `kpack` 是 CDNA 专属参数,RDNA 不需要,却参与 name hashing,使搜索空间最多膨胀 4 倍;
2. 缺少基于寄存器溢出(register spilling)的剪枝,坏配置可导致底层 kernel 崩溃;
3. Triton 3.6+ 上 `triton.runtime.cache` 是没有 `.clear()` 的模块,既有缓存清理逻辑失效。

## 修复方案
修改 `benchmarks/kernels/benchmark_moe.py`:
1. `get_rocm_tuning_space`:仅 `use_fp16 and on_cdna()` 时加入 `matrix_instr_nonkdim`/`kpack` 参数空间,RDNA 不再搜索这两个维度;
2. `prune_rocm_configs`:RDNA(gfx11/gfx12)按每 wave 累加器 VGPR 数剪枝,`accum_vgprs >= 256` 的配置被拒绝(WMMA tile 为 16,每 tile 累加器占 8 个 VGPR);`matrix_instr_nonkdim` 默认取 `WMMA_TILE=16`;
3. `clear_triton_cache`:Triton 3.6+ 走 else 分支,遍历 `gc.get_objects()` 清理每个 `JITFunction.device_caches`;
4. `tune` 循环新增 `except RuntimeError` 捕获,打日志并 `continue` 跳过坏配置,避免 worker 挂死。

```diff
-    if use_fp16:
-        param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
-        param_ranges["kpack"] = kpack_range
+    # RDNA doesn't use matrix_instr_nonkdim nor kpack
+    if use_fp16 and on_cdna():
+        param_ranges["matrix_instr_nonkdim"] = [16, 32]
+        param_ranges["kpack"] = [1, 2]
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本号。
- 回归面:仅 `benchmarks/kernels/benchmark_moe.py` 调参脚本;CDNA 的调参空间与剪枝逻辑不变,RDNA 的搜索空间显著缩小(正文称最多缩减 4 倍);基于 VGPR 的剪枝公式假设 MoE 恒以 fp32/int32 累加且 wave32 模式,若假设不成立可能误杀可行配置(由 diff 推断);`RuntimeError` 捕获会把部分失败静默降级为 warning,可能掩盖真实错误(由 diff 推断)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
