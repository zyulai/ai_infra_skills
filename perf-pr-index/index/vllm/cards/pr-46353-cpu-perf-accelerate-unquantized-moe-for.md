# PR #46353: [CPU][Perf] Accelerate unquantized MoE for AArch64

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-06-24 |
| 分类 | MoE |
| Label | performance, ready, ci/build, cpu, gpt-oss |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 12+ 个文件,另有 5 个文件未展示 |
| 链接 | https://github.com/vllm-project/vllm/pull/46353 |

## 一句话结论

在 AArch64 上启用 CPU FusedMoE 并新增 AdvSIMD BFMMLA 微内核加速 w13/w2 GEMM,gpt-oss 吞吐 2.1x、gemma4 2.34x(96 核 Neoverse-V2)。

## 动机

CPU 后端的 unquantized fused MoE 路径(`cpu_fused_moe`/`cpu_prepack_moe_weight`)此前只在 `__AVX512F__` 下编译注册,AArch64 完全没有可用的 MoE kernel;MicroGemm 框架与 `swigluoai_and_mul` 等激活函数也是 x86/AVX 专属(sleef.h、gather load)。

## 优化点

四层改动。第一,新增 `cpu_micro_gemm_neon.hpp`:基于 `vbfmmlaq_f32`(BFMMLA,K=4)的 BFloat16 MicroGemm 特化,8x8 与 4x16 两种 micro tile(BF16 权重,Nr=8,`WeightOCGroupSize=Nr`),提供按物理布局 `[M/8, K/4, 4, 2, 4]` 重排的 `pack_weight` 与 `pack_input_from_rows`。第二,把 MicroGemm 接口泛化:各 ISA 实现新增 `WeightOCGroupSize` 与 `PackA` 两个 constexpr,`fused_moe_impl` 在 `PackA=true` 时改为把输入行 pack 到专用 buffer(w13 输入与 w2 输入各一次,并按 packed input 调整 L2 cache 预算与 w2 n-tile 大小),GEMM 调用去掉 lda 依赖;weight group stride 由硬编码 16 改为 `gemm_t::WeightOCGroupSize`。第三,`swigluoai_and_mul` 在 aarch64 下用新的 `load_even_odd`(`vuzpq_f32` 解交织)替代 AVX gather load,并把 sleef 的 `tanhf16_u10` 抽象成 `FP32Vec16::tanh()` 方法(x86 用 Sleef,ARM 标量回退用 `std::tanh`,gelu_tanh 激活受益)。第四,torch bindings 的 `cpu_fused_moe`/`prepack_moe_weight` 注册条件扩为 `__AVX512F__ || ARM_BF16_SUPPORT`,benchmark 脚本 ISA 选项增加 "neon"。

## 关键改动

```text
csrc/cpu/micro_gemm/cpu_micro_gemm_neon.hpp: 新增 BFMMLA BFloat16 MicroGemm(pack_weight/pack_input_from_rows/8x8 与 4x16 micro kernel)
csrc/cpu/cpu_fused_moe.cpp: fused_moe_impl 支持 PackA 输入打包与 WeightOCGroupSize 抽象;ARM 激活路径
csrc/cpu/cpu_types_arm.hpp: FP32Vec16 增加 load_even_odd/取负等 AdvSIMD 原语
csrc/cpu/{cpu_types_x86.hpp,cpu_types_scalar.hpp}: tanh() 统一接口(Sleef 封装进 x86 类)
csrc/cpu/torch_bindings.cpp: cpu_fused_moe/prepack_moe_weight 放开 ARM_BF16_SUPPORT 编译
```

## 性能声明

96 Neoverse-V2 cores,`vllm bench throughput --num-prompts 256 --dataset-name sharegpt(ShareGPT_256_seed0.json) --output-len 256 --max-model-len 4096 --max-num-batched-tokens 4096 --tensor-parallel-size 1 --no-enable-prefix-caching`:unsloth/gpt-oss-20b-BF16 吞吐提升 2.1x,google/gemma-4-26B-A4B-it 提升 2.34x。未给出基线绝对数值;gemma4 需 `VLLM_CPU_ATTN_SPLIT_KV=0` 以规避 attention 挂起(与本 PR 无关的既有问题)。

## 验证与回归风险

- Test Plan 为 CI(diff 中未见新增 pytest 文件,benchmark 脚本 `benchmark_cpu_fused_moe.py` 更新支持 neon ISA);x86 路径的 PackA=false/WeightOCGroupSize=16 与原实现等价,行为应不变。
- 风险:NEON MicroGemm 仅特化 BFloat16(其他标量类型直接 TORCH_CHECK 失败),要求 K 与 output_size 分别被 4/8(及 NSize=32)整除;BFMMLA 需要 `ARM_BF16_SUPPORT` 硬件(FEAT_BF16),老 ARM 核上不可用;`load_even_odd` 假定 gate/up 交错布局输入连续;pack_input 引入额外内存拷贝,在小 M 场景可能得不偿失(无独立数据)。

## 关联

- 同区域:#24701(CPU attn py run SDPA 重构)、CPU 后端的量化 MoE 支持;ISA 分派机制同 AMX/VEC 两条既有路径。
- 受益模型:gpt-oss(交织 gate-up 权重)、gemma4(gelu_tanh)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
