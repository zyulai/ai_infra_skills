# PR #56045: [CPU][Perf] refactor paged attention for Arm CPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-09-20 |
| 分类 | Attention / Kernel \| tier: major |
| Label | cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 2 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/56045 |

## 一句话结论

重构 Arm CPU paged attention，paged attention 最高快 25%，并删约 100 行代码。

## 动机

PR 正文 Purpose 仅写 "Previous implementation left performance on the table."（此前实现未充分释放性能），随后列出本次提供的改进点，未进一步展开改动前的具体瓶颈细节。

## 优化点

正文列出：更快的 GEMM kernels、更快的 kv cache packing、在拷贝时预打包 Q、新增接口允许 attention impl 把 softmax probabilities 直接写成 packed format、对 block_size % 16 启用 "NEON" ISA、paged attention 最高快 25%，并整体删除约 100 行代码。diff 层面，csrc/cpu/cpu_attn_impl.hpp 引入 ProbabilityTokenStore 模板与 probability_token_stride，让概率缓冲按可配置 stride 写入；csrc/cpu/cpu_attn_neon.hpp 把 BLOCK_SIZE_ALIGNMENT 从 32 改为 16；cpu_attn_neon_bfmmla.hpp 同步适配。其中 AttentionMainLoop 改用 probability_token_store_t 与 probability_token_stride 替换原先对 curr_prob_buffer_iter 直接按 16 递增的写法，使 packed A 的写入步长可配置；cpu_attn_neon_bfmmla.hpp 亦同步适配新的对齐与存储接口。

## 关键改动

```text
csrc/cpu/cpu_attn_impl.hpp: 新增 ProbabilityTokenStore / probability_token_stride，概率按 packed 格式存储
csrc/cpu/cpu_attn_neon.hpp: BLOCK_SIZE_ALIGNMENT 由 32 改为 16，block_size % 16 启用 NEON
csrc/cpu/cpu_attn_neon_bfmmla.hpp: bfmmla 实现同步适配
```

```cpp
// This allows us to store probabilities in a packed format
// for GEMM implementations that need packed A.
template <typename attention_impl_t, typename = void>
struct ProbabilityTokenStore {
  using prob_buffer_t = typename attention_impl_t::prob_buffer_t;
  static constexpr int32_t TokenStride = 1;

  FORCE_INLINE static void store_probabilities(
      prob_buffer_t* __restrict__ probability, const vec_op::FP32Vec16& values,
      const int32_t, const int64_t) {
    using prob_buffer_vec_t = typename VecTypeTrait<prob_buffer_t>::vec_t;
    prob_buffer_vec_t output_vec(values);
    output_vec.save(probability);
  }
};
```

## 性能声明

PR 正文只给出 "up to 25% faster paged attention"（perf claims 亦检出 "up to 25%" / "25% faster"），未提供 benchmark 表格或逐项数字明细；PR 未注明测试条件（模型 / 硬件 / 精度 / workload 均未说明）。

## 验证与回归风险

Test Plan 仅为 CI；Test Result 正文留空（PR 未给出具体测试结果）。回归风险正文未具体说明；从 diff 看改动涉及 NEON 对齐（BLOCK_SIZE_ALIGNMENT 由 32 降为 16）、packed probability 存储格式与 probability_token_stride，可能影响 Arm CPU 上不同 block_size 的 paged attention 与 GEMM 概率缓冲路径。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
