# PR #34167: [DSA] Fix top-k v2 dropping non-primary ranks' output on CUDA 13.1+ (root cause for #33835)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-08-10 |
| 分类 | Sampler / Detokenization (tier: major) |
| Label | run-ci, jit-kernel, bypass-fastfail |
| Diff | pack 未提供 diffstat（diff 内容可用：4 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/34167 |

## 一句话结论

修复 CUDA 13.1+ 编译器下 DSA 小批量 cluster top-k 静默丢弃 7 个非 primary rank 输出的 bug（#33835 的非法内存访问根因），顺带 fused-cluster 快 1.3-2.9%。

## 动机

DeepSeek-V4 DSA decode 会从 fused small-batch cluster kernel 返回损坏的 top-k，下游 sparse attention 把垃圾 KV 索引解引用即 #33835 报告的 illegal memory access。触发条件是 CUDA toolkit 版本而非 GPU 架构：同一份源码、同一块 H200（sm_90a），nvcc 12.9/13.0 正确、13.1/13.2/13.3 出错。

## 优化点

根因：`TopKCluster::forward` 在 phase-3 提前把两个 scatter 目的地合成一个指针变量（`cur_out = is_primary ? problem.out : smem->tmp_out`）；`problem.out` 可能是 elected rank buffer 的 `shared::cluster`（DSMEM）别名，`tmp_out` 是 `shared::cta`，单指针携带两种地址空间使 cicc 13.1+ 在 sm_90a 上错误 lower block-local 分支，非 primary rank 的暂存输出全部静默变零，phase 3.5 再忠实地把零拷去正确的 DSMEM 地址。修复为把 primary 与非 primary 拆成两条独立代码路径（各自独立循环），并加注释防止再被合并；同时删掉 #32910 的 `peer_problem` 拷贝，改在读回处 `__builtin_assume(problem.out == topk_indices)`（这是防止 #32830 cicc 段错误的承重断言，删掉会在 13.1+ 复现崩溃）。第二项独立改动：top-k v1 的 `topk` 从 `-DSGL_TOPK` 编译期宏改为运行时参数（`kMaxTopK=1024`）——旧机制每个 k 编一个模块，宏喂 `constexpr` 导致所有模块 mangled symbol 相同，函数局部 static 被 `STB_GNU_UNIQUE` 跨模块合并，第二个加载的模块跳过 `cudaFuncSetAttribute` 后以 64KB dynamic smem 启动失败。

## 关键改动

```text
python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh: phase-3 拆分 primary/非 primary 两条 scatter 路径（实际修复）
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh: 删 peer_problem 拷贝，改承重 __builtin_assume
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v1.cuh: topk 改运行时参数，kTopKBlockSize 固定 1024
python/sglang/kernels/ops/attention/dsv4/topk.py: _jit_topk_v1_module 不再按 k 分别编译
```

## 性能声明

无性能回归（CUDA 12.9 下新旧都正确，等正确性对比；单位 us）：fused-cluster 8x33000 8.387 -> 8.230、8x98304 9.431 -> 9.309、30x131072 13.805 -> 13.564、1 长 + 7 短（report shape）9.740 -> 9.575，fused cluster 路径快 1.3-2.9%；register/streaming/persistent 路径持平（最差 +0.1%，噪声）。

## 验证与回归风险

H200 157 行测试套件（report shapes + 边界扫描 + 控制），k in {512,1024,2048}：nvcc 12.9/13.0 前后 0 bad；13.1/13.2/13.3 修复前 157 bad、修复后 0 bad。另 1500 次随机 stress、500 次 CUDA graph replay 干净；sm_90a 与 sm_100a 全工具链编译通过；v1 runtime-topk 路径 80 cases（k 覆盖 1..1024）对 trivial/boundary/radix 形状全过。#33835 的边界回归测试保留。范围修正：#33835 归因于 Hopper 与 32K 阈值上方的行，实测所有 fused small-batch cluster 形状（batch<=30 且 seq_len>cluster_floor）都受影响（含 #33835 报告为正常的 65537/98304），persistent-pool 路径从未受影响。风险：sm_100a 仅编译验证、无 Blackwell 运行时确认；CI 用 cu130（干净组合），此类 bug 现有 CI 抓不到；v1 遗留的 exported-weak-symbol 隐患（kernel host stub）未解决。

## 关联

#33835（症状报告，本 PR 为其根因）；#32830（cicc 段错误）、#32910（peer_problem 拷贝 workaround，被本 PR 移除）；#33835 所加边界回归测试被保留。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
