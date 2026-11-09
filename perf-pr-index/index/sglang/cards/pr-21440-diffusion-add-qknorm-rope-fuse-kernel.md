# PR #21440: [Diffusion] Add qknorm rope fuse kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-03-27 |
| 分类 | Attention / Kernel (major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 含 6+ 个文件(JIT CUDA kernel、runtime helper、4 个 DiT 接入、测试与基准),另有 2 个文件因体积从 pack 截断 |
| 链接 | https://github.com/sgl-project/sglang/pull/21440 |

## 一句话结论

新增 QK RMSNorm + RoPE 融合的 warp 级 in-place JIT CUDA kernel,微基准 1.44x,Qwen-Image denoise 阶段快 14.35%。

## 动机

diffusion DiT 里大量存在"QK norm 后接 RoPE"的 pattern,但此前 Q/K RMSNorm 走 jit_kernel 的 qk norm、RoPE 走 FlashInfer 实现,两次独立调用:Q、K 各多一次全局内存往返,且两个 kernel launch 分开。基于 AKO4ALL 框架(Codex 生成)把这个 pattern 融合为单一 kernel。

## 优化点

新增 `qknorm_rope.cuh` 的 warp 级融合 kernel:每个 warp 处理一个 (token, head) 工作项,输入与 RMSNorm 权重用向量化 packed load;RMSNorm 在 warp 内用 warp-level reduction 完成(不用 shared memory,fp32 累加保证数值稳定),归一化值留在寄存器里立即被 RoPE 消费——标准布局做 pairwise 旋转,NeoX 布局用 `__shfl_xor_sync` 做 lane 交换,结果 packed 后 in-place 写回。合并 Q、K 处理到一次 launch,消除 split 路径在 QKNorm 与 RoPE 之间的一次全局内存往返;按 `head_dim`/`rope_dim`/`is_neox`/dtype 做 JIT 特化与 occupancy-aware launch sizing,支持 PDL(`PDLWaitPrimary`/`PDLTriggerSecondary`)。runtime 层新增共享入口 `apply_qk_norm_rope`/`apply_qk_norm_with_optional_rope`,支持 `position_offset` 分段位置(FLUX/FLUX.2 双流 attention 中 text 与 image token 用不同 RoPE 位置范围);默认启用(env `SGLANG_ENABLE_FUSED_QKNORM_ROPE=1`),不满足形状/dtype/layout 条件时回退原 split 路径(jit qknorm + FlashInfer RoPE)。接入 qwen_image、flux、flux_2、zimage 四个 DiT。

## 关键改动

```text
python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh: warp 级融合 kernel(head_dim≤256、必须整除 warp)
python/sglang/jit_kernel/diffusion/qknorm_rope.py: JIT 加载、can_use 守卫(head_dim 64/128/256、rope_dim 约束)、custom op 注册
python/sglang/multimodal_gen/runtime/layers/layernorm.py: apply_qk_norm_rope 共享入口 + position_offset + 回退
python/sglang/multimodal_gen/runtime/models/dits/{qwen_image,flux,flux_2,zimage}.py: 融合路径接入
python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py 与 benchmark/diffusion/bench_qknorm_rope.py: 正确性与微基准
```

NeoX lane 交换核心:

```cuda
float swapped = __shfl_xor_sync(kActiveMask, elems[i], kHalfRotaryLanes);
if (lane_id < kHalfRotaryLanes) { swapped = -swapped; }
elems[i] = elems[i] * cos + swapped * sin;
```

## 性能声明

- 微基准(split = JIT qknorm + FlashInfer RoPE vs fused,5 个 case:flux_1024、qwen_image_1024/partial、zimage_1024、batch2_medium,形状 [B·T, H, D],D=128):各 case 1.3760x-1.4753x,加权 **1.4387x**。
- E2E denoise 阶段(`sglang generate`,1024x1024,seed 42,`--enable-torch-compile`,各模型开启 warmup;Qwen-Image 50 步 CFG4、Z-Image 9 步等):qwen 14.43→12.36 s(**−14.35%**,1.1675x);qwen-edit −1.26%、flux −1.14%、flux2 −0.01%、zimage −1.47%。模型:Qwen-Image-2512、Qwen-Image-Edit-2511、FLUX.1-dev、FLUX.2-dev、Z-Image-Turbo。
- PR 正文的 benchmark 部分未注明 GPU 型号(仅测试代码注释提及 H200 的容差背景)。

## 验证与回归风险

新增 `test_qknorm_rope.py`(CI 注册 est 44s + nightly):batch×heads×head_dim(64/128/256)×is_neox×position dtype(int32/int64)×rope_dim 组合,与 split 路径比对(atol 8e-2/rtol 1e-2,split 基线混用 BF16 qknorm 与 FlashInfer RoPE,与融合路径相差约一个 BF16 舍入步)。CI 状态在 pack 中未展示。风险面:融合要求 q/k 4D 同形、contiguous、q_eps==k_eps、weight dtype 一致;NeoX 要求 rotary lane 数为 2 的幂;不满足时回退 split;qwen_image 路径额外校验 image_rotary_emb 必须是 cos_sin_cache 张量。flux_2 中顺带清理了未使用的 lora_scale 变量。

## 关联

前序:#17241(Flux2Attention 接入 apply_qk_norm,本 PR 在其上把 RoPE 一并融合)。同区域:diffusion JIT kernel 系列(#40405 Wan 后处理融合、#29281 Conv3D cat-pad 等,pack 未确认相互关联)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
