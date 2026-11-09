# PR #12348: [ROCm] Faster Custom Paged Attention kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tjtanaa |
| 合并 | 2025-03-03 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, rocm, frontend, speculative-decoding, ready, ci/build |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),另有 5 个文件因超长从 diff 中省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/12348 |

## 一句话结论

用 `mfma16x16x16` 指令重写 ROCm Custom Paged Attention kernel,MI300X 上 Llama-3.1-70B fp8 端到端输出吞吐提升约 15%-54%。

## 动机

ROCm 上的 Custom Paged Attention(v2 路径)原 kernel 基于 `mfma4x4x4` 指令与 512 的 partition 粒度,单条 MFMA 只处理 4x4 的 QK 片段,decode 密集场景下 attention kernel 成为瓶颈。该功能移植自 ROCM/vllm fork(PR #372),目标是利用 CDNA3(MI300 系列)的 16x16x16 MFMA 指令与 FP8 转换内建指令提高 QK^T 与 Softmax·V 的计算吞吐。

## 优化点

新增 `paged_attention_ll4mi_QKV_mfma16_kernel`:每条 `gcn_mfma16x16x16_instr`(f16 或 bf16_1k 变体)处理 16 个 query head x 16 个 token 的 QK 片段(Q head 与 token 维度跨 lane 组织,每个 lane 持 16B 连续 head elements),Softmax·V 阶段同样以 16x16x16 MFMA 处理。grid 为 `(num_seqs, num_partitions, num_kv_heads)`、block 256、token partition 固定 256,`__launch_bounds__(NUM_THREADS, 5)` 提高占用。FP8 KV cache 路径用 MI300+ 的 `v_cvt_pk_f32_fp8`(2 条指令转换 4 个 fp8;gfx90a 走 4 指令 fallback)先解包再转回 16bit 做 MFMA;`from_floatx4`/`addx4` 等辅助操作改用 packed half2/bf16 位操作替代逐元素循环。原 kernel 保留并更名为 `mfma4` 变体作对照。注:`LOGITS_RTZ_CONVERSION` 常量置为 false(正文注明因对精度的影响禁用 RTZ 转换)。

## 关键改动

```text
csrc/rocm/attention.cu: 新增 paged_attention_ll4mi_QKV_mfma16_kernel(16x16x16 MFMA 主路径),原 kernel 更名为 mfma4 变体;新增 fp8 packed 转换 helper(to_float_fp8x4/convert_b8x8_custom)
benchmarks/kernels/benchmark_paged_attention.py: 新增 --custom-paged-attn 开关与 ROCm partition size 256
csrc 与 python 侧 dispatch/注册(在省略文件中): ops.paged_attention_rocm 入口
```

关键指令封装:

```cpp
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid, blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz, cbid, blgp);
  }
}
```

## 性能声明

端到端(`benchmark_serving`,Llama-3.1-70B-Instruct,TP1,MI300X,KV-cache fp8_e4m3,fp8 量化):128 输入/128 输出 200 prompts:Req/s 13.05 → 15.11,输出 tok/s 1670.3 → 1933.87;128/2048:Req/s 0.78 → 1.20(+54%),输出 tok/s 1598.45 → 2455.57;512/128:输出 tok/s 824.51 → 938.81;512/2048:输出 tok/s 1357.31 → 2118.84;ShareGPT 1000 prompts:Req/s 6.22 → 7.45,输出 tok/s 1234.64 → 1477.79。全部 7 组负载输出吞吐提升约 15%-54%,总 token 吞吐同步提升(如 128/2048 总 tok/s 1698.35 → 2609.04)。PR 未单独给出 kernel 级微基准数字,也未注明 baseline 与优化两侧的运行次数/方差。

## 验证与回归风险

改动携带 kernel benchmark 扩展(`--custom-paged-attn` 开关、ROCm partition 256),并来自已在 ROCM/vllm fork(PR #372)验证的实现;标签含 ci/build。风险面:新 kernel 为模板特化(HEAD_SIZE/GQA_RATIO/BLOCK_SIZE/ALIBI/dtype 组合),覆盖不全的形状需回退 mfma4 变体;FP8 路径依赖 MI300+ 的 `v_cvt_pk_f32_fp8` 内建指令(gfx90a 有单独 fallback 分支);RTZ 转换被禁用说明精度敏感位已保守处理。正文未附单元测试清单,精度对比数据未在正文中给出。

## 关联

同区域:ROCm attention 后续演进见 pr-39616(AITER MLA attention)、pr-50578(MLA asm decode)、pr-34709(wvSplitK skinny GEMM);上游 fork 来源 ROCM/vllm#372。同分类(Attention / Kernel)其他 PR 见索引。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
