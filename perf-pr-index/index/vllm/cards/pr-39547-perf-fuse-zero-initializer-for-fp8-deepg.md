# PR #39547: [Perf] Fuse Zero Initializer for FP8 DeepGemm Block Quant Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-04-11 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 2 个文件(csrc 量化 kernel 与对应测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/39547 |

## 一句话结论

将 FP8 DeepGemm 块量化的 scale 清零融合进 kernel,MiniMax-M2.5 解码约 1% E2E 加速。

## 动机

PR 正文 Purpose 指出:`per_token_group_quant_fp8_packed_for_deepgemm` 之前必须先调用 `torch::stable::zero_(output_s_packed)` 初始化 scale buffer,再执行量化 kernel;因为该 kernel 用 `atomicOr` 把各 group 的 UE8M0 exponent 累加进同一个 32-bit 字,必须先清零。这次额外的 zero_ 是一次独立 kernel launch 与显存写,正文测得它在解码层时间中占 2 * 1.2 us(240 us 层时间),值得消除。

## 优化点

PR 去掉 host 侧的零初始化,改为在量化 kernel 内部对 scale buffer 的 padded 索引直接写 0。kernel 签名新增 `num_groups_padded`、`padded_groups_per_row`、`num_scale_elems` 等参数,把 grid 从覆盖 `num_groups` 扩展为覆盖 `tma_aligned_mn * padded_groups_per_row`,使 output_s_packed 的每个字节都被写到。线程用 `(mn_idx, sf_k_idx)` 二维映射判断是否为有效 group:有效组仍计算 scale,但把 `atomicOr` 累加改为按字节直接赋值(`reinterpret_cast<uint8_t*>`);无效(padding)组在边界内写 0。host 侧同时新增对 output_s_packed stride 必须为 `[1, tma_aligned_mn]` 的 TMA 对齐校验,并删除 `torch::stable::zero_(output_s_packed)` 调用。

## 关键改动

```text
csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu: 核心量化 kernel,扩展 grid 覆盖 padding、kernel 内写零替代 atomicOr,host 侧移除 zero_ 并新增 stride 校验
tests/kernels/quantization/test_per_token_group_quant.py: 新增 per_token_group_quant_fp8_packed_for_deepgemm 的 kernel 测试(含 packed scale storage 逐字节校验)
```

关键机制代码(padding 组写零、有效组直接按字节写 exponent):

```cpp
    if (is_valid_group) {
      const unsigned int bits = __float_as_uint(y_s);
      const uint8_t exponent = static_cast<uint8_t>((bits >> 23u) & 0xffu);
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = exponent;
    } else if (out_idx < num_scale_elems) {
      // write zero for padding groups if within bounds of output_s_packed
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = 0;
    }
```

## 性能声明

PR 正文:对 Minimax M2.5 Fp8、concurrency 128、1K/1K,省去 2 * 1.2 us out of 240 us layer time,即 **~1% E2E speedup**(解码)。正文未注明硬件型号与精度参数。Test Result 给出 gsm8k 精度对比:`vllm serve MiniMaxAI/MiniMax-M2.5 --trust-remote-code --tensor-parallel-size 2` 下 exact_match 0.9295(flexible)/0.9265(strict),`VLLM_USE_DEEP_GEMM=1` 下 0.9325/0.9265;作者据此做 E2E 模型精度 eval 说明无回归。

## 验证与回归风险

Test Plan:新增 `tests/kernels/quantization/test_per_token_group_quant.py` 中的 kernel 测试,并做 MiniMax 2.5 的 E2E 模型精度 eval。影响路径:所有走 `per_token_group_quant_fp8_packed_for_deepgemm`(FP8 DeepGemm 块量化)的解码/量化路径;新增的 stride 校验会拒绝非 TMA-aligned 的 output_s_packed,属于潜在兼容性约束。正文未注明其他硬件或模型的影响范围。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
