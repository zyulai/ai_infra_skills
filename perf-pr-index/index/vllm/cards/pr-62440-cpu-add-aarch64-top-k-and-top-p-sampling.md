# PR #62440: [CPU] Add AArch64 top-k and top-p sampling kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @micyan01 |
| 合并 | 2026-11-19 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ci/build, cpu, verified |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及新增 benchmarks/kernels/cpu/benchmark_cpu_topk_topp.py 与 csrc/cpu/topk_topp_sampler.cpp(813 行)等,另有 7 个文件被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/62440 |

## 一句话结论

把 #43452/#55874 的 CPU top-k/top-p 采样 C++ 内核(AArch64 NEON + 标量实现)清理后落到当前 main,Neoverse-V2 上 top-p/组合过滤对 PyTorch 排序实现提速 10–15x,端到端 Qwen3-0.6B 解码吞吐最高 +14.62%。

## 动机

v1 CPU 采样的 top-k/top-p 过滤走 PyTorch 排序实现,在 131k 词表上单行 top-p 耗时 10.5 ms 以上;此前两个 PR(#43452 基础 C++ 内核与集成,Tianmu Li;#55874 AArch64 实现,Fadi Arafeh)未进入 main。本清理提交移除 AVX512/AVX2 采样路径与 `-mfma` 构建标志,保留 AArch64 与标量实现,并把 `TestCpuTopkTopp` 加进 Arm CPU 硬件 CI kernel shard。

## 优化点

1. 新 `csrc/cpu/topk_topp_sampler.cpp`:ternary search 求 top-k pivot、binary search 求 top-p 边界,`NORMAL_CDF_TO_SIGMA_TABLE` / `PERCENTILE_TO_STD_TABLE`(200 项,与 `topk_topp_triton.py` 一致)做离群预筛,边界并列值(`kBoundaryTol`/`kKDuplicateTol`)按行序保留。
2. AArch64 NEON 向量化:`Fp64Acc16` 用 FP64 累加 128K 元素的 softmax 求和(避免 FP32 掉 ~3 ULP),`sum_gt_to_double` / `count_within_tol` / `mask_write_below` / `vec_max_min_with_pad_blend` 全部 16-lane 向量化,pad 哨兵 `-1e30f` 混入 `+inf` 保护 reduce_min;PowerPC/S390(VSX/VXE)同样有 `vec_doublee` 路径。
3. 与 PyTorch 基线的 18 个配置输出 mask 完全一致;输入 clone 不计入测量区。

## 关键改动

```cpp
// csrc/cpu/topk_topp_sampler.cpp(AArch64 FP64 累加器)
+struct Fp64Acc16 {
+  float64x2_t acc[8] = {vdupq_n_f64(0.0), ...};
+  inline void add(const vec_op::FP32Vec16& v) { ... vcvt_f64_f32 ... }
+  inline double reduce() const { ... vaddvq_f64(sum); }
+};
```

```cpp
+static float ternary_search_topk(const float* buf, int n, int k, float lo,
+                                 float hi, float* dup_value_out,
+                                 int* num_dup_out, int* num_above_out)
```

## 性能声明

数字均照抄 PR 正文(未独立核实)。条件:AWS Neoverse-V2,96 物理核无 SMT;FP32 Zipf-like logits(`rank^-1.1`),词表 131,072;batch 1/8/64;top-k k=50、top-p p=0.9、组合 k=100/p=0.9;5 次预热 + 20 次测量;3 次 A/B 交替运行。

- Operator(节选):1 线程 batch 1:top-k 0.466→0.315 ms (1.48x),top-p 10.537→0.995 (10.59x),top-k+top-p 10.665→0.742 (14.38x);1 线程 batch 64:top-k 21.919→19.117 (1.15x),top-p 672.106→62.797 (10.70x),组合 681.255→45.613 (14.94x);16 线程 batch 64:top-k 2.163→1.323 (1.64x),top-p 43.061→4.062 (10.60x),组合 43.917→3.206 (13.70x)。
- End to end(`Qwen/Qwen3-0.6B`, BF16, 128/128 tokens, 32 requests/并发级, temperature=0.6/top_k=50/top_p=0.9/min_p=0, prefix caching disabled, EOS ignored):c1 吞吐 8.20→8.90 tok/s (+8.48%),TPOT 120.60→111.07 ms (−7.90%);c4 30.64→33.56 (+9.53%);c8 59.02→64.60 (+9.46%);c16 100.63→115.35 (+14.62%),TPOT 149.07→128.84 (−13.57%),E2E reduction 12.76%。

## 验证与回归风险

- 干净 CPU 源码构建于 AArch64 Neoverse-V2:`TestCpuTopkTopp` 100 tests passed;CPU fused sampling suite 35 tests passed;适用 pre-commit hooks 通过;AI assistance 在 commit metadata 中披露。
- 回归风险:仅保留 AArch64 与标量路径,x86 不再从此 PR 获得加速(原有 AVX 路径被移除);top-p 边界并列处理依赖 1e-5 容差,与 Triton 参考的并列决策需保持一致;服务器端到端数字来自同一 PR 源码与二进制、仅切换 PyTorch/C++ 过滤实现的 A/B。

## 关联

来源:#43452(基础 C++ 内核与集成)、#55874(AArch64 实现);对照实现 `apply_top_k_top_p_pytorch` 与 GPU 侧 `topk_topp_triton.py`。索引内同区域:CPU 采样与 FlashInfer 采样卡(如 #11377/#15777 一类,pack 未提供精确映射)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
