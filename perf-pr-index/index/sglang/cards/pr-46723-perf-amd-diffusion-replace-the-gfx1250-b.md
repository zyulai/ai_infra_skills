# PR #46723: [Perf] [AMD] diffusion: Replace the gfx1250 bf16 scale-shift torch fallback with a bit-exact Triton kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yichiche |
| 合并 | 2026-10-29 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/46723 |

## 一句话结论

gfx1250 上用 bit-exact Triton kernel 取代 bf16 scale-shift 的 torch fallback,该 kernel 快 6.4x。

## 动机

#42859 因现有 Triton kernel 在 gfx1250 miscompile(恒等项被丢弃),把所有 bf16 `fuse_scale_shift_kernel` 调用路由到 torch fallback。当 scale 为 fp32 时,fallback 全程以 fp32 计算并返回 fp32,Wan 的 `mlp_residual(...).to(orig_dtype)` 因此要对 [90000, 5120] 张量做三次遍历:fp32 broadcast mul、fp32 add、bf16 cast,共 2.18 ms/block,每次 forward 40 个 block。

## 优化点

新增 `_scale_shift_fp32_rows_kernel`:按 16x256 row tile 分块、int64 row 偏移,支持 per-token 或 broadcast shift,单 kernel 完成 `x * (scale_constant + scale) + shift` 并一次性舍入到 `x.dtype`。bit-exact 的关键是复现 fallback 的算术:同样的 fp32 中间运算、`enable_fp_fusion=False` 禁用 FMA 收缩、只在最后舍入一次,结果与 fallback 后接 `.to(x.dtype)` 完全一致(也正是所有调用方的下一步)。入口 `_fuse_scale_shift_fp32_gfx1250` 先做 guard(3-D x、fp32 per-batch scale、unit stride on C、shift 形状/dtype),不满足返回 None 继续走 fallback;kernel 在现有 `is_gfx1250_supported() and x.dtype is torch.bfloat16` 分支内被优先尝试。bf16 scale、per-frame 4-D scale、非 3-D x 等场景不受影响。

## 关键改动

```text
python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py: 新增 fp32-scale 行分块 Triton kernel 与 gfx1250 guard,接入 fuse_scale_shift_kernel 分发
test/registered/kernels/ops/diffusion/test_modulate.py: gfx1250-only 的 torch.equal bit-exact 测试
```

```python
out = x * (scale_constant + scale)[None, :] + shift
tl.store(out_ptr + ..., out.to(out_ptr.dtype.element_ty), mask=mask)
# 启动参数:BLOCK_L=16, BLOCK_C=256, num_warps=4, enable_fp_fusion=False
```

## 性能声明

- Kernel microbenchmark,gfx1250,[1, 90000, 5120] bf16 x、fp32 scale、bf16 shift:torch fallback + `.to(bf16)`(main)2.176 ms → 本 PR 0.341 ms(6.4x)。
- 端到端,`sglang generate` Wan2.2-T2V-A14B 720x1280、193 frames、8 steps、CFG、单 gfx1250 partition、单请求(`--num-gpus 1 --attention-backend aiter`、`SGLANG_USE_ROCM_FLYDSL=1`、`HIP_FORCE_DEV_KERNARG=0`):denoise 199.3 s → 197.3 s,e2e 223.3 s → 221.5 s。
- 精度:端到端输出视频与 main bit-identical(逐帧 PSNR = inf,max abs diff 0)。

## 验证与回归风险

新增 gfx1250-only 测试以 `torch.equal` 对比 `fuse_scale_shift_kernel_native(...).to(bf16)`,覆盖 k in {0, 1}、full/broadcast shift、bf16/fp32 shift 共 8 个参数化用例,另有 24 个 shape/dtype 用例(L = 777/90000/176400,C = 5120)。注意点:PR 指出 `test_scale_shift_3d_tiling_is_bit_exact`(#46147 加入)在 main 的 gfx1250 上本就 8/8 失败(其参考是被绕开的 miscompiled `blc_opt` kernel),本 PR 后 3/8 通过,剩余用例失败是与 miscompiled 参考比较所致。影响范围限于 gfx1250 + bf16 输入;其他平台不进入该分支。CI 的 AMD ROCm 10 run 在合并时为失败状态。

## 关联

#42859:将 gfx1250 bf16 scale-shift 路由到 torch fallback 的前序 PR;#46147:引入 `test_scale_shift_3d_tiling_is_bit_exact` 的相关 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
