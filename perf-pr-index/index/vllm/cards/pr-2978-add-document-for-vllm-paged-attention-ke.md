# PR #2978: Add document for vllm paged attention kernel.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pian13131 |
| 合并 | 2024-03-04 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数;diff 显示新增 7 张图与 2 个文档文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/2978 |

## 一句话结论

为 vLLM paged attention CUDA kernel(`csrc/attention/attention_kernels.cu`)撰写开发者文档,讲解内存布局、读取模式与逐步计算流程;纯文档 PR,无代码改动。

## 动机

Paged attention kernel 是 vLLM v0 吞吐能力的核心,但其实现对新人相当复杂:KV cache 按物理 block 分页存储、kernel 用特制的内存布局与访存模式(尤其 global→shared memory 的搬运)换取性能,缺少面向实现层面的解释材料。作者在学习该 kernel 后撰写了高层文档,意图帮助后续阅读实际代码。

## 优化点

无代码优化。文档(`docs/source/dev/kernel/paged_attention.rst`,525 行,配 7 张示意图)按 kernel 执行顺序组织:输入与模板参数(`HEAD_SIZE`/`BLOCK_SIZE`/`PARTITION_SIZE` 等);核心概念(Sequence、Context、Vec、Thread group、Block、Warp、Thread block、Grid 的分工映射);Query 段讲 `q_vecs` 在 shared memory 中按 thread group 行分布以获得相邻线程访存合并(coalescing);Key 段讲 `k_ptr` 按 block/head/token 寻址与 `k_vecs` 放寄存器的原因(单线程单次访问);QK 段讲 `Qk_dot<>::dot` 的跨 thread group 归约;Softmax 段讲 `qk_max`/`exp_sum` 的 warp 内 `shfl_xor` 与 thread block 两级归约;Value/LV 段讲 v_vec 与 logits_vec 的点积累积、`accs` 的 warp 内归约与跨 warp shared memory 折半合并;最后是输出写回。明确声明不覆盖索引计算与点乘实现细节。

## 关键改动

```text
docs/source/dev/kernel/paged_attention.rst: 新增 525 行 kernel 解析文档
docs/source/assets/kernel/*.png: 新增 7 张示意图(query/key/value/q_vecs/k_vecs/logits_vec/v_vec)
docs/source/index.rst: Developer Documentation 目录加入该文档
```

## 性能声明

无性能声明(纯文档)。文档中与性能相关的表述为定性解释:q/k 的 vec 划分使每次取 16 字节、相邻线程读相邻内存以实现 coalescing;k 数据放寄存器而 q 数据放 shared memory 的理由;value 沿列(token)读取以配合 `V_VEC_SIZE` 的点积模式。均为对既有 kernel 设计的解释,非本 PR 引入的优化。

## 验证与回归风险

无测试(文档 PR,未触及代码)。风险为零:仅新增 rst 与图片资产、修改 docs index。

## 关联

同区域:v0 paged attention kernel 及其后续优化卡(如 pr-29 memcpy kernel for FlashAttention、pr-12348 ROCm custom paged attention、pr-29193 CPU paged attention GEMM 等)。该文档描述的 `attention_kernels.cu` 布局是理解 v0 attention 优化的基础材料。
