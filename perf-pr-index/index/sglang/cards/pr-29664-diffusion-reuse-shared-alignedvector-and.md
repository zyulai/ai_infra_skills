# PR #29664: [Diffusion] Reuse shared AlignedVector and tidy jit_kernel/diffusion

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-30 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(diff 完整可见,改 10 个文件:.cuh 3 个 + CuTe-DSL/Triton 清理) |
| 链接 | https://github.com/sgl-project/sglang/pull/29664 |

## 一句话结论

diffusion CUDA kernel 的手写 union 统一替换为共享 `device::AlignedVector`,B200 上 SASS/ncu 级别等价,行为不变的性能卫生型改动。

## 动机

KDA-Pilot diffusion 原生 CUDA 快路径都需要 128-bit 向量化 load/store。最先落地的 `norm_scale_shift`(#27392)已使用 `sgl_kernel/vec.cuh` 的共享 `device::AlignedVector` 组件,但两个较新的 kernel(#29281、#29361)仍各自手写本地 union。本 PR 让所有 diffusion CUDA kernel 复用共享 AlignedVector 而非 per-kernel union,并顺手合入审查 `jit_kernel/diffusion/` 时发现的一批已验证的小清理。声明无行为改变——这是工程质量型 PR,不追求性能增益。

## 优化点

Native CUDA 三个 kernel 换向量组件:`residual_gate_add` 删除本地 `union Vec16<T>` 改用 `AlignedVector<T,kVec>` 的 load/store/operator[];`causal_conv3d_cat_pad` 的 `union Pack` 换为 `AlignedVector<ET,kVec>`(散乱标量读保持 `__ldg`,只取向量化 store);`timestep_embedding` 的裸 `float4` 换 `AlignedVector<float,4>`,并补 `#pragma once` 与具名 namespace `sglang_timestep_embedding`(包装符号同步更新)。CuTe-DSL/Triton 清理:把两份逐字节相同的 `to_cute_arg`/`to_fake_cute_args` 助手提升到 `cutedsl/utils.py`(去掉约 36 行重复);修复 `norm_tanh_mul_add_norm_scale` 中 copy-paste 的 guard(`isinstance(src,...) and isinstance(src,...)` → 第二个应为 `dst`);删除 `sana_wm_gdn` 已被 `fused_qk_inv_rms` 取代的死代码 `_precompute_inv_rms`;4D `scale_shift` kernel 与调用点删除未用的 `rows` 参数;`validate_weight_bias` 去掉未用的 `B, S` 参数;清理陈旧注释代码与 docstring。

## 关键改动

```text
python/sglang/jit_kernel/csrc/diffusion/residual_gate_add.cuh: union Vec16 -> device::AlignedVector
python/sglang/jit_kernel/csrc/diffusion/causal_conv3d_cat_pad.cuh: union Pack -> AlignedVector
python/sglang/jit_kernel/csrc/diffusion/timestep_embedding.cuh: float4 -> AlignedVector<float,4>,具名 namespace
python/sglang/jit_kernel/diffusion/cutedsl/utils.py: 提升共享 to_cute_arg/to_fake_cute_args
其余 6 个文件:重复删除、guard 修复、死代码与无用参数清理
```

```cuda
using Vec = device::AlignedVector<T, kVec>;
...
Vec r, u, g, o;
r.load(residual, v);
u.load(update, v);
g.load(gate, v);
for (int i = 0; i < kVec; ++i) {
  o[i] = residual_gate_value(r[i], u[i], g[i]);
}
o.store(out, v);
```

## 性能声明

验证平台 B200,OLD=union/float4,NEW=AlignedVector(registered bench 的 cuda 列,中位数):

- `bench_residual_gate_add`(6 个 workload):中性,如 ltx2_bcast_s32640_c4096 126.35 → 125.97 µs、flux2_bcast_s512_c3072 15.93 → 15.68 µs
- `bench_causal_conv3d_cat_pad`(4 个 case):噪声内(<1% 摆动)
- 为排除真实回归做的 SASS diff(最大 shape):指令数相同(1184 vs 1184),两者都是 1x STG(单条 128-bit store)+ 8x LDG、0 条 local-memory 操作(无寄存器溢出),唯一差异是一处 `IMAD`↔`IADD3` 调度交换
- `ncu` 佐证:DRAM 吞吐相同(约 18.5%)、SM 吞吐相同(约 81%)、local load/store sectors 均为零;该 kernel 是 SM-issue-bound,亚 1% 差异为编译调度/测量噪声
- `timestep_embedding` 无 registered bench,仅单测覆盖(仍是单条 128-bit store)

## 验证与回归风险

B200 上 pytest 全过:`test/registered/jit/diffusion/{test_residual_gate_add, test_causal_conv3d_cat_pad, test_fused_norm_scale_shift, test_qwen_image_modulation}.py` 与 `test/registered/jit/test_timestep_embedding.py`;另对所有被触碰 kernel 做 torch 参考对齐 smoke(4 个 CuTe-DSL modulation 入口、4D scale_shift、`sana_wm_gdn.fused_qk_inv_rms`,外加 `validate_weight_bias` 拒绝坏 shape 的负测试),bf16 容差内匹配。回归面:timestep_embedding 的 JIT 包装符号改为带 namespace 的全名;`validate_weight_bias` 签名变化需调用点同步(已在 diff 内);norm_tanh guard 修复使 copy_if 正确判断 dst 类型。

## 关联

前序:#27392(norm_scale_shift,首个 AlignedVector 使用者)、#29281、#29361(被统一的手写 union kernel)。后续:PR #36521(4D scale-shift autotune,同目录 modulate 模块)。同区域为 KDA-Pilot diffusion native-CUDA 快路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
