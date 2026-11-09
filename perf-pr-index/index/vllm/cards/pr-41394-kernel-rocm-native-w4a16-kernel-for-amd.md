# PR #41394: [Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JartX |
| 合并 | 2026-05-29 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, ci/build, v1 |
| Diff | diffstat 缺失(素材未提供官方统计);PR 正文自述 "12 files changed, +2820 / 0 lines" |
| 链接 | https://github.com/vllm-project/vllm/pull/41394 |

## 一句话结论

为 RDNA3 (gfx1100) 增加原生 HIP W4A16 kernel,bf16 快 2.5-4.2x 消除 ExLlama fp16-only 的精度/速度二选一。

## 动机

RDNA3 上此前唯一的快速 W4A16 路径是 ExLlama,但它是 fp16-only:Qwen2/Qwen3、Llama 3.x、Mistral 等现代模型以 bf16 训练发布,权重范围/激活幅度常超 fp16 的 ±65504 动态范围,cast 到 fp16 会产生 inf/NaN 传播、静默劣化与工具调用输出损坏。Triton W4A16 虽支持 bf16,但约 82 tk/s,比 ExLlama fp16 慢约 3x。用户被迫在"快但数值错"与"对但慢到不可用"之间取舍。本 PR 用原生 bf16 W4A16(205-345 tk/s)消除该取舍,fp16 路径同时比 ExLlama 快 6-17%。另可加载 AWQ 权重(如 cyankiwi/Qwen3.5-9B-AWQ 系列)。

## 优化点

内核由 exllamav2 的 4-bit kernel 改造而来,两条计算路径共用一个 `torch.ops._C.gptq_gemm_rdna3` 入口(C++ 侧分发,保持 `apply_weights` torch.compile 友好):(1) 标量点积 kernel 服务 decode 与 M<16——fp16 用 `v_dot2_f32_f16`(`__builtin_amdgcn_fdot2`,hipcc 不会自动 peephole 出该指令,显式发出可省每 dot22_8 调用 3 个转换 op);bf16 因 gfx11 缺 `v_pk_fma_bf16` 而显式 widen 到 fp32 用 `v_fma_f32` 累加,M=1 时走 `v_dot2_f32_bf16` + BF16_MAGIC(0x43004300)魔数反量化(20 v_dot2 + 8 FMA vs 原 40 FMA,约省一半),并用 fp32 别名 union 阻止 InstCombine 折叠(LVM #76000);M_COUNT∈{1,2,4,8} 按 size_m 启动期选择。(2) `v_wmma_f32_16x16x16` kernel 服务 bf16 M≥16(M≥64 时 v2 比 v1 省 5-7%):v1 单 wave 16Mx16N tile,v2 双 wave 32Mx16N + LDS 双缓冲。K 维经 gridDim.z 切分后,输出用 64-bit CAS 环路模拟 packed atomic add(gfx11 无 `v_global_atomic_pk_add_{f16,bf16}`)直接写入 T 类型输出,免去 fp32 scratch buffer + memset + cast pass;BLOCK_KN_SIZE 从 exllama 的 128 提到 256(减半每输出位置的 atomic 数;试过 512 但 fp16 decode 回退 40% 而回退)。kernel 注册为 `RDNA3W4A16LinearKernel`,排在 ROCm 优先级列表 Triton kernel 之前,`can_implement()` 以 `on_gfx11()` 门控,CDNA/MI3xx/RDNA4 自动回落 TritonW4A16。

## 关键改动

```text
csrc/rocm/q_gemm_rdna3.cu: 标量点积 W4A16 kernel(decode/M<16)+ CAS packed atomic + M=1 bf16 快路径
csrc/rocm/q_gemm_rdna3_wmma.cu: v_wmma_f32_16x16x16 prefill kernel(v1/v2,独立 TU 防 hipcc TU 级优化误编译)
csrc/rocm/qdq_4_rdna3.cuh: RDNA3 反量化(bf16 用专用 bit-trick 避免 fp16 上半 nibble*16 技巧的 7-bit 尾数溢出)
csrc/quantization/gptq/README_RDNA3.md: 架构说明、fragment layout、已回退实验与 bench 命令
```

## 性能声明

端到端 evalscope perf(openqa 数据集,50 请求、concurrency 100、输出约 1800 tok),vLLM serve Qwen3.6-27B-GPTQ-W4A16-G32 于 RX 7900 XTX、tp=2、`--kv-cache-dtype fp8 --enable-prefix-caching`:Triton W4A16 bf16 @max-num-seqs=8 为 82.37 tk/s(基线),fp16 83.23;ExLlama fp16 255.00 (seqs=8) / 382.52 (seqs=32);本 PR bf16 205.34 (seqs=8) / 345.64 (seqs=32),即较 Triton bf16 快 2.49x/4.20x;本 PR fp16 270.19 (seqs=8) / 445.74 (seqs=32),较 Triton 快 3.28x/5.41x,较 ExLlama fp16 +5.96%/+16.5%。bf16 对 ExLlama fp16 慢 9.6-19.5%(ExLlama 无 bf16,仅作量化代价参考)。TPOT:fp16 seqs=8 为 0.028 s。核级 M-sweep:WMMA bf16 在 M≥16 胜出,v2 在 M≥64 比 v1 省 5-7%;fp16 WMMA 端到端与 scalar 打平(445.7 vs 440.7 tk/s,尽管微基准有 47% 优势),故 fp16 保持 scalar 分发。

## 验证与回归风险

测试计划(全部勾选):gfx1100 容器构建;`gptq_gemm_rdna3_wmma_probe` op 验证 mode-1 fragment layout;v1/v2 kernel 在 {M=32..2048, K=N∈{4096,5120,11008}, fp16/bf16} 下 parity 到数值舍入级;M-sweep 微基准(`bench_prefill_msweep.py`);禁用本 kernel(`VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel`)复现 ExLlama fp16 基线,确认 decode-only 无回退;VGPR/spill 审计:bf16 M_COUNT=8 从 192-VGPR 上限 + 144 spill 降到 130 VGPR / 0 spill,WMMA v1/v2 均 43-44 VGPR。风险面:仅 gfx11 门控,其他 ROCm 架构走原 Triton 路径;要求 size_n % 8 == 0(64-bit CAS 对齐)、group_size >= 32(与 exllama 假设一致);scalar 与 WMMA 分拆两个 TU 以避免 hipcc TU 级寄存器启发式导致 M=1 路径静默误编译(作者实测踩坑);rocprof 在 gfx1100 多数计数器返回 0,诊断深度受限。

## 关联

同区域:RDNA3/RDNA 线性路径(#40835 int8/int4 KV-cache HIP kernel 分支被引用;#40977 wvSplitK_int4 hybrid kernel 的数据流被 M=1 路径镜像);kernel 生态补齐见 ExLlama/TritonW4A16/Conch 等 linear kernel 优先级列表。后续工作:JartX 分支 feat/rdna3_int8_int4_hip_kernels 与 perf/rdna3_full_stack。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
