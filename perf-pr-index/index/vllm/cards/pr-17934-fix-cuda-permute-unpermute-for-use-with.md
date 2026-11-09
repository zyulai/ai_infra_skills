# PR #17934: Fix CUDA permute/unpermute for use with DeepGemm Moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @CalebDu |
| 合并 | 2025-07-27 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(diff 全文已提供但裁剪了 1 个文件,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/17934 |

## 一句话结论

重构 DeepGemm MoE 用的 CUDA permute/unpermute kernel:去掉 `token_expert_indices` 输入依赖、输出对齐 triton kernel,并修复 clamp 导致 argsort 结果错误的 bug。

## 动机

`deep_gemm_moe.py` 调用 CUDA 版 `moe_permute/moe_unpermute` 时存在两类问题:一是接口依赖冗余输入 `token_expert_indices` 且输出语义与 triton kernel 不一致,参数中还有未使用项;二是(引用 deep_gemm_moe.py L84-86,cc @bnellnm)clamp 处理会导致 argsort 结果错误,使 permute 产生错误映射。该 kernel 负责把 token 按 expert 分组展开(permute)与 unpermute+reduce,正确性直接影响 DeepGemm grouped GEMM 路径。

## 优化点

接口重构:`moe_permute` 不再接收 `topk_weights`(quant scale 改为可选 `a1q_scale`,按 `permuted_idx.clamp(...) // topk` 重排),内部用 `torch.arange` 自建 `token_expert_indices` 判断序;EP 场景下先 clone `topk_ids` 再做 preprocess,避免原地修改调用方的 topk_ids;输出从 `src_row_id2dst_row_id_map` 换成 `inv_permuted_idx` + 新增 `permuted_idx`(hidden→permuted_hidden 映射),与 triton kernel 输出对齐。kernel 修正:`expandInputRowsKernel` 的 source row 计算 `expanded_source_row % num_rows` 改为 `expanded_source_row / k`,并写入 `permuted_idx`(跳过非本地 expert);`finalizeMoeRoutingKernel` 删除 `expert_for_source_row` 参数,展开行索引由 `original_row + k_idx*num_rows` 改为 `original_row*k + k_idx`;修复 `getMIndicesKernel` 的 shared memory 索引按 tidx 而非 i 写入的越界/错位 bug。Python 侧 `moe_permute` 返回五元组,`moe_unpermute` 改为输出参数 + `inv_permuted_idx`,`expert_first_token_offset` 变 optional。

## 关键改动

```text
csrc/moe/moe_permute_unpermute_op.cu: moe_permute/moe_unpermute 签名重构;topk_ids clone 防 EP 原地污染
csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl: source_row 除法修正;permuted_idx 写入;finalize 索引修正;getMIndices smem 索引修复
csrc/moe/torch_bindings.cpp: schema 更新
vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py: python wrapper 对齐(a1q_scale 重排、返回 inv_permuted_idx/permuted_idx)
(deep_gemm_moe.py 的 clamp 修复在素材 diff 中被裁剪)
```

索引修正节选:

```cuda
// 旧:int64_t const source_row = expanded_source_row % num_rows;
int64_t const source_row = expanded_source_row / k;
...
// 旧:expanded_original_row = original_row + k_idx * num_rows;
int64_t const expanded_original_row = original_row * k + k_idx;
```

## 性能声明

PR 正文未提供性能数字(作者写"will attach performance data in H100 later",素材中未见);本 PR 以正确性修复与接口对齐为主。GPU/workload 测试条件无。

## 验证与回归风险

`tests/kernels/moe/test_moe_permute_unpermute.py` 重写 torch 参考实现并扩展参数(NUM_EXPERTS 增至 256,含 EP 1/4/16、align_block_size 两分支):逐项断言 expert_first_token_offset、inv_permuted_idx、m_indices、有效行 permuted_hidden 与 unpermute 输出。`benchmark_moe_permute_unpermute.py` 适配新签名。风险面:接口变更是 breaking 性质,所有 CUDA permute/unpermute 调用方(deep_gemm_moe 等)需同步更新;EP 场景 topk_ids clone 增加一次小拷贝。

## 关联

同区域:DeepGemm MoE 路径(deep_gemm_moe.py)与 permute/unpermute kernel 系列(#31837 为 NVFP4 cutlass 侧同类 launch 优化);具体编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
