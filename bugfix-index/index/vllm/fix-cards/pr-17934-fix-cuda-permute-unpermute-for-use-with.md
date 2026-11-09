# PR #17934: Fix CUDA permute/unpermute for use with DeepGemm Moe

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17934](https://github.com/vllm-project/vllm/pull/17934) |
| 作者 | CalebDu |
| 合并日期 | 2025-07-27 |
| labels | performance, ready |
| 热度 | 👍 4 · 💬 44 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
重构 CUDA MoE permute/unpermute kernel 以适配 DeepGemm:去掉 token_expert_indices 依赖、输出对齐 triton kernel、修 clamp 导致 argsort 结果错误。

## 问题现象
正文未描述具体现象,仅说明改动动机:该 CUDA permute/unpermute kernel 要与 DeepGemm MoE 搭配使用,存在三处问题——[1] kernel 依赖外部传入的 `token_expert_indices` 且输出与 triton kernel 不对齐、参数冗余;[3] deep_gemm_moe.py L84-86 处的 clamp 会导致 argsort 结果错误(作者 @bnellnm)。性能数据(H100)作者称稍后附上,正文中未见。

## 根因
正文披露:[3] 的 bug 是 clamp 造成 argsort 结果错误(正文链接到 `vllm/model_executor/layers/fused_moe/deep_gemm_moe.py#L84-L86`,未展开细节);[1] 的问题是对 `token_expert_indices` 输入的依赖与输出和 triton kernel 不对齐。diff 另可见两处索引错误的修正:`expandInputRowsKernel` 中 `source_row` 由 `expanded_source_row % num_rows` 改为 `/ k`,`finalizeMoeRoutingKernel` 中 `expanded_original_row` 由 `original_row + k_idx * num_rows` 改为 `original_row * k + k_idx`,以及 `getMIndicesKernel` 共享内存写入由 `smem_...[tidx]` 改为 `[i]`(由 diff 推断,正文未逐条说明)。

## 修复方案
1. **重构 kernel 接口**(`csrc/moe/moe_permute_unpermute_op.cu`、`torch_bindings.cpp`、`moe_permute_unpermute.py`):`moe_permute` 去掉 `topk_weights` 输入、`token_expert_indices` 改为内部 `torch.arange` 生成;`src_row_id2dst_row_id_map` 换成 `inv_permuted_idx` + 新增 `permuted_idx` 输出;`moe_unpermute` 精简为 `(out, permuted_hidden_states, topk_weights, inv_permuted_idx, expert_first_token_offset?)` 且改为写 out 参数;preprocess 用 `topk_ids.clone()` 避免原地改写调用方张量;Python 层新增 `a1q_scale` 按 `permuted_idx` 重排返回。关键 diff 摘录:
```cuda
-    int64_t const source_row = expanded_source_row % num_rows;
+    int64_t const source_row = expanded_source_row / k;
```
2. **修 clamp/argsort 错误**与输出对齐 triton kernel(即上述索引公式修正)。
3. 移除未用参数与 `finalizeMoeRoutingKernel` 的 `expert_for_source_row` 输入;benchmark 脚本同步更新。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响使用 CUDA permute/unpermute + DeepGemm MoE 的路径。
- 回归面:`csrc/moe/`(kernel、op 封装、torch schema)与 `vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py` 的 Python API——`moe_permute`/`moe_unpermute` 签名与返回值均变化,调用方(含 `deep_gemm_moe.py`、benchmark)需同步适配;[2]「把 kernel 集成进 deepgemm moe」在正文中被划掉,集成不在本 PR 内。性能数据作者承诺 H100 后补,正文未见,无法评估性能回归。

## 关联
- 正文提到的讨论对象:@bnellnm(_CLAMP 相关代码位置 deep_gemm_moe.py#L84-L86);未列出其他 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
