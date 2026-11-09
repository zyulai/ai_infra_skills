# PR #936: [BugFix] Fix NaN errors in paged attention kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#936](https://github.com/vllm-project/vllm/pull/936) |
| 作者 | WoosukKwon |
| 合并日期 | 2023-09-04 |
| labels | (none) |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
paged attention 对 context 外 token 计算 0×NaN=NaN 污染输出;在最后一个 value block 显式清零越界 value。

## 问题现象
正文描述:paged attention kernel 会对 value block 内所有 token 计算 `attn_weight * value`,即使其中部分 token 并不在 context 内。由于这些 token 的 attn_weight 为 0,通常没有影响;但当这些 token 的 value 含 NaN 时,`0 * NaN = NaN` 使结果出错。对应 issue #641。

## 根因
正文披露:kernel 对 value block 内全部 token(含 context 外的 token)做点积;"attn_weight 为 0 即可忽略"的假设只在 value 为有限值时成立,token 的 value 含 NaN 时 0×NaN 仍为 NaN,污染累加器。

## 修复方案
`csrc/attention/attention_kernels.cu` 的 `single_query_cached_kv_attention_kernel` 中,对最后一个 block(`block_idx == num_blocks - 1`)把超出 `context_len` 的 token 的 value 显式置零:

```diff
+        if (block_idx == num_blocks - 1) {
+          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
+          // we should explicitly zero out the values since they may contain NaNs.
+          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
+#pragma unroll
+          for (int j = 0; j <= V_VEC_SIZE; j++) {
+            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
+          }
+        }
```

同时在 `dtype_bfloat16.cuh` 新增 `zero(__nv_bfloat16&)`(`__CUDA_ARCH__ < 800` 时 assert,否则按位写 0x0000),在 `dtype_float32.cuh` 新增 `zero(float&)`,并把 `dtype_float16.cuh` 中的 `zero(uint16_t&)` 移到文件末尾以匹配新的调用顺序。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围。
- 回归面:paged attention CUDA kernel 最后一个 value block 的路径(仅该 block 增加逐 token 边界比较);新增 bf16 `zero()` 在算力 <800 下 assert(与 bf16 硬件要求一致);fp16/fp32 的 `zero()` 为纯新增辅助函数。

## 关联
- #641(正文 "Fixes #641")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
