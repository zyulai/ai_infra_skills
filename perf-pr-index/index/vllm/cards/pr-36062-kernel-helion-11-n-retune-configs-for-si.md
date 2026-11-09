# PR #36062: [Kernel] [Helion] [11/N] Retune configs for silu_mul_fp8

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gmagogsfm |
| 合并 | 2026-03-05 |
| 分类 | Quantization(major) |
| Label | ready |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 仅含 1 个 JSON 配置文件,另有 1 个文件因篇幅被省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/36062 |

## 一句话结论

用 "full" 档 autotune 重调 silu_mul_fp8 的 Helion 配置,H200 上各 intermediate size 内核 geomean 提速 1.29x–2.31x。

## 动机

此前 silu_mul_fp8 的 tuned 配置由 "quick" 档 autotune 生成,且对小的 intermediate_size/num_tokens 组合不够有效。本 PR 是 Helion 内核配置重调系列的第 11 步,只重调配置、不改内核代码。

## 优化点

在 H200 上以 "full" effort 重跑 autotune,再对小 intermediate_size/num_tokens 手工微调配置。diff 显示大量 per-(intermediate, num_tokens) 条目的搜索空间选择被整体替换:block_sizes、num_warps/num_stages、l2_groupings、flatten_loops、load eviction policy、indexing(pointer vs tensor_descriptor)与 pid_type(flat/xyz/persistent_interleaved)。典型变化:numtokens_1 的多数条目改为 flatten_loops=true 并加大第二维 block(如 32→256);多个 numtokens_256 条目 num_warps 从 1/2 提到 8;个别大条目改用 xyz/persistent_interleaved 调度并附 num_sm_multiplier/maxnreg。

## 关键改动

```text
vllm/kernels/helion/configs/silu_mul_fp8.json: 覆盖 intermediate 2048–14336 × num_tokens 1–64
  的 Helion autotune 配置全量重调(唯一改动文件,纯数据变更)
```

## 性能声明

H200 上按 intermediate size 的 geomean speedup(括号内 min/max):2048: 1.29x (1.11x–1.74x);2880: 1.42x (1.06x–1.93x);4096: 1.50x (1.04x–2.21x);8192: 1.84x (1.05x–3.16x);11008: 2.07x (1.17x–4.00x);14336: 2.31x (1.22x–4.88x)。PR 未注明 dtype 与具体 workload(仅说明为 silu_mul_fp8 内核级对比)。

## 验证与回归风险

无新增测试;纯配置变更,风险限于按 (intermediate_size, num_tokens) 查表命中 silu_mul_fp8 的调用路径。配置只在 H200 上重调,其他 GPU 命中同表不保证最优,但可按 size 粒度回退。

## 关联

Helion 内核配置 retune 系列([11/N]);pack 未列出前序/后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
