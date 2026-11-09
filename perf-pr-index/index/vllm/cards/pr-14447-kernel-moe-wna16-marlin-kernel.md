# PR #14447: [Kernel] moe wna16 marlin kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2025-04-15 |
| 分类 | MoE (tier: major) |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 含 csrc/moe/marlin_moe_wna16/ 新目录(generate_kernels.py、kernel.h、marlin_template.h 等),另有 14 个文件因超长未含于 pack |
| 链接 | https://github.com/vllm-project/vllm/pull/14447 |

## 一句话结论

新增基于 marlin m-parallel 机制的 MoE wna16(GPTQ/AWQ)kernel,DeepSeek-V3-AWQ 8xA800 生成吞吐最多提升约 1.5-2.7x。

## 动机

PR #12185 与 #13321 引入了 triton/cuda 的 MoE wna16 kernel 用于优化 GPTQ/AWQ MoE 推理,但当时性能最好的 GPTQ/AWQ kernel 是 marlin kernel。已有 moe+marlin 的实现没有充分发挥 marlin 的性能优势,尤其在大 expert 数场景下。若直接把 marlin 用于 MoE,所有 MoE block 并行处理会带来过高的 `workspace` 与 `c_tmp` 容量需求,这是需要解决的工程问题。

## 优化点

核心机制是利用 marlin kernel 固有的 m-parallel 特性并行处理所有 MoE block:每个 threadblock 处理一个 stripe,跨多个 moe block(以 `sorted_token_ids`/`expert_ids`/`num_tokens_past_padded` 组织)迭代。关键改进在 workspace 逻辑——由于不同 SM 共享的 `slice_col_par` 最大不超过 SM 数量,workspace 只需固定长度等于 SM 数,从而避免了 moe block 数量增长带来的 workspace/c_tmp 容量爆炸。kernel 基于 vllm 的 gptq_marlin 实现,完整继承其特性(bfloat16/int8/act_order 等,weight 类型覆盖 kU4/kU4B8/kU8B128),并支持 expert parallelism(`is_ep` 分支跳过 expert_ids 为 -1 的无效 block)。kernel 模板通过 jinja2 脚本按 dtype/位宽/group_blocks/thread 配置组合批量生成,以控制 wheel 体积。

## 关键改动

```text
csrc/moe/marlin_moe_wna16/marlin_template.h: 核心 Marlin MoE kernel 模板(m-parallel 处理 moe block、atomic add/fp32 reduce 输出归约)
csrc/moe/marlin_moe_wna16/generate_kernels.py: jinja2 批量生成各模板实例化的 kernel_*.cu 文件
csrc/moe/marlin_moe_wna16/kernel.h: kernel 参数与模板声明(MARLIN_KERNEL_PARAMS)
```

关键代码摘录(m-parallel 与 workspace 定容逻辑,节选,<=15 行):

```cuda
// parallel: num valid moe blocks
int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
int parallel = num_tokens_past_padded / moe_block_size;
...
int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);
```

## 性能声明

- 注:PR 正文注明以下 kernel benchmark 为早期(过时)结果,最终结果在 PR #16850 的评论中(main 段)。
- kernel benchmark(A800,DeepSeek-V3-AWQ TP=8 / Qwen1.5-MoE-A2.7B-GPTQ-Int4 TP=1 / Mixtral-8x7B-AWQ TP=1 的 shapes,gist 数据为图片,具体数字未在 pack 文本中给出);结论:新 kernel 优于旧 marlin kernel;k 较小时 marlin 表现较差(bit=8 时甚至慢于 triton 版),原因是 marlin 为大 n 大 k 设计。
- DeepSeek-V3-AWQ,8xA800,`VLLM_MARLIN_USE_ATOMIC_ADD=1`(单位疑为 tok/s,PR 表格未标明):
  - generation-bs=1:main 42.7 → PR 45.9(MLA)/ 49.9 → 54.4(no-MLA);bs=32:381.9 → 558.5;bs=256:806 → 2046.7;bs=512:1392 → 2754.4(MLA);no-MLA bs=512:1461.7 → 2991。
  - prefill(MLA-main 4735.7 → MLA-PR 7986.5)。
- 准确性:DeepSeek-R1-AWQ 8 卡 lm_eval gsm8k(5-shot):flexible-extract exact_match 0.9560,strict-match 0.9568。

## 验证与回归风险

- 准确性验证依赖 lm_eval gsm8k 对 DeepSeek-R1-AWQ 的实测;kernel 由模板生成器批量实例化,覆盖 fp16/bf16 x 3 种 weight 类型 x act_order/zp/group_blocks 组合。
- 影响路径:GPTQ/AWQ 量化的 MoE 模型走 fused MoE 的 kernel 选择逻辑;k 较小的 shape 上 marlin 可能不如 triton 版本(选择策略需依赖 tuning);`VLLM_MARLIN_USE_ATOMIC_ADD` 为可选开关;pack 未显示新增单测文件。

## 关联

- 前序:PR #12185、PR #13321(triton/cuda moe wna16 kernel);最终性能对比在 PR #16850 评论中。
- 同区域:索引库已有 pr-16850(some optimizations for dense marlin)、pr-34130(fused moe int4 w4a16 benchmark)等 MoE 量化 kernel 系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
