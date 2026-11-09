# PR #30924: [JIT] Trait-driven per_token_group_quant: unify the quant kernel family (flat + masked)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-07-22 |
| 分类 | Quantization(major) |
| Label | documentation, quant, run-ci, jit-kernel, bypass-fastfail |
| Diff | pack 未提供 diffstat(diff 正文可用,另有 15 个文件因过长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/30924 |

## 一句话结论

用一个 trait 驱动的 JIT kernel 收编整个 per-token-group 量化 kernel 家族,dense 网格普遍提速 29-41%,masked EP-MoE 场景最高 16.4x。

## 动机

per-token-group 量化在代码库中积累了大量平行实现:sgl-kernel AOT v1/v2、JIT v1/v2、`dsv4/fp8_wo_a`、`minimax_quant_ue8m0`、多个 Triton 变体(含已死亡的 `_hopper_moe_mn_major`)、五路 runtime dispatcher 加 `SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT` 环境开关。维护面大且旧 kernel 留有性能空间(如 v2 对 masked EP-MoE 在 padded token 轴上发固定大小 grid,小 batch 时 35 us 起步)。本 PR 依 RFC #29630 引入单一 JIT kernel 作为唯一的 CUDA group-quant 路径并完成全部调用点迁移。

## 优化点

把量化的每个属性变成单个 `QuantTrait` kernel body 的编译期模板轴:输入 dtype(bf16/fp16)、输出 dtype(fp8_e4m3/int8)、group size(16-256 模板轴,v2 只运行时切换且上限 128)、scale 格式(fp32 / 4-per-int32 packed UE8M0,含 pack-tail 清零)、scale 布局(row-major / TMA 对齐 col-major)、融合激活(none / silu_and_mul)、调度(flat 2D / masked EP-MoE 3D)。机制上:每个 group 由一个 subwarp(`group_size/vec_size` lane)处理,16B(Blackwell 32B)向量化读写,absmax 在 packed 16-bit 域用 #30838 的 `warp::reduce_max` 归约,无跨 warp 通信;masked 调度在 token 轴 grid-stride 到设备端 `masked_m[e]`,`expected_m` 仅作 launch 大小提示,提示错误不会丢 token;两种调度均支持 PDL,`masked_m` 接受 int32/int64 而不需第二份实例化(读低 32 位字)。迁移要点:`_run_per_token_group_quant_8bit_kernel` 收敛为 MUSA→AOT v2、CUDA→新 kernel 两路;whole-row group quant 改走专用 `sgl_per_token_quant_fp8`;`_varlen_deep_gemm_silu_mul_quant` 直接按 deep_gemm 布局发射 scale(省掉 get_mn_major 变换),移除 `SGLANG_MASKED_GEMM_FAST_ACT` 分支;SM100+ 上 fp32 scale 乘法用 `__fmul2_rn`(16 FMUL → 8 FMUL2)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/gemm/per_token_group_quant.cuh: 新增 QuantTrait 模板 kernel(flat + masked 两个调度)
python/sglang/jit_kernel/per_token_group_quant.py: JIT 包装与 KernelSpec 注册的 custom op
python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit.cuh: 删除旧 JIT v1 kernel
(迁移触及 quantization dispatcher、int8 路径、_varlen_deep_gemm_silu_mul_quant 等)
```

## 性能声明

B200 上与 v2 JIT kernel 对比(随 PR 附 `bench_per_token_group_quant{,_masked}.py`)。Flat(hidden=4096,col-major UE8M0,group 128):2048 tokens 5.778 → 4.562 us(4067 → 5151 GB/s),1/32 tokens 基本持平。Masked EP-MoE(DeepSeek-V3 shape,4 GPU,balanced routing):1 token 35.50 → 2.17 us(16.4x),128 tokens 35.48 → 2.93 us(12.1x)。PR 声明新 kernel 在 dense 网格带宽饱和的 batch 上普遍 ≥ v2,典型 29-41% 更快;`_varlen_deep_gemm_silu_mul_quant` 路径在 balanced 负载下比 DSV4 JIT activation kernel 快 25-60%(B200)。SM100+ FMUL2 改动带来大 batch 下 +0.5-1%。

## 验证与回归风险

新增 `test/registered/jit/test_per_token_group_quant.py`(92 例,注册 1-gpu-large 与 4-gpu-b200):UE8M0 路径(DeepGEMM/EP-MoE 生产路径)对纯 torch 参照 bit-exact(含 hidden=768 非 4 对齐 pack-tail 的 row/col-major);fp32/int8 scale 路径锁定存储 scale bit-exact 并约束反量化误差;masked 调度验证写入行 bit-exact、padding 行不被触碰、`expected_m` 欠提示不丢 token。参照全部用纯 torch 而非 v1/v2 kernel,测试不依赖被替换的 kernel 存活。作者注明既有问题(非本 PR 引入):`test_v2_jit_matches_aot` 在 CI 子集之外 66 例失败。风险面:int8 路径 codes 相对旧 AOT-v1 在约 0.2% 元素上偏移 ≤1(fast-math 除法边界平局),scale bit-exact;显式排除范围(不改)包括 DSV4 `swiglu_limit`/swizzled gran=8 kernel、DSV4 `fp8_wo_a`、ROCm/MUSA/XPU Triton 回退。

## 关联

Stacked on #30838(DTypeTrait/warp reduction 基建,前 4 个 commit 属于它);依 RFC #29630 迁移,rebase 于 #30784 之后。后续 PR 计划:Triton `_packed_fwd` 加 activation-kind 轴、删除 v2 JIT kernel 与死亡 Triton 变体、折叠 `minimax_quant_ue8m0`。同区域:#41725 等涉及量化 kernel 的 PR。
