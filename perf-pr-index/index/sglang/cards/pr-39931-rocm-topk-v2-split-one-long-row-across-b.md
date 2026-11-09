# PR #39931: [ROCm] topk v2: split one long row across blocks, the CDNA cluster-path equivalent

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @EricKing626 |
| 合并 | 2026-10-04 |
| 分类 | Sampler / Detokenization (tier: major) |
| Label | run-ci, jit-kernel, bypass-fastfail |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/39931 |

## 一句话结论

ROCm top-k v2 新增长行跨块拆分(两阶段 global-memory 直方图),补齐 CDNA 无 cluster/DSMEM 的缺口,小批量长上下文 kernel 提速 1.8–3.0x,600K 上下文 ITL 降 18.7%。

## 动机

top-k v2 默认每行一个 block,推理模型 decode 常见的小批量长上下文场景下(batch 8、128K 也只用 8 个 block),单行由一个 block 扫描,256-CU 的 GPU 上大部分 CU 闲置。NVIDIA Hopper+ 上超过长度阈值的行会拆到共享 DSMEM 的 block cluster;CDNA 没有 cluster 和 DSMEM,因此 ROCm 无法支持 cluster 路径。#36851 使 topk_v2 成为 GLM 在 ROCm 上的默认实现后,该缺口成为主要瓶颈。

## 优化点

用两次 launch 加 global memory 替代 DSMEM:每行拆 `clamp(64/R, 4, 32)` 个 block(目标每次 launch 约 64 个 block;再多的跨块开销随 R*split 增长,抵消扫描收益)。第一阶段 `topk_split_hist` 各 block 对自己分片建立 12-bit 直方图(chunk 起点对齐整 wavefront 的 16B 向量加载);第二阶段 `topk_split_select` 各 block 从合并直方图求阈值 bin 并追加候选,LDS 暂存后每 block 一次全局 atomic 提交,最后一个到达的 block(经 `SplitCounters` 上的 fence/atomic arrival 选举)解 tie 并应用 page-table transform,同时把直方图清零使 scratch 可单流复用。触发条件:batch ≤ 64(实现上 R ≤ CU/4)且行长超过随 batch 提高的下限(R≤8 为 40960,R>32 为 114688);过短行由 rank 0 走原单块路径。scratch 按最坏情况一次性分配约 2 MB/卡(HIP graph 把地址烤进 kernel 参数,不能释放)。CUDA 不受影响(无 `USE_ROCM` 时预处理 TU 与 baseline 逐字节相同);prefill/ragged/packed 不进新路径。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh: 新增 TopKSplit 两阶段 kernel、split_plan 门控与 SplitResources scratch
test/registered/kernels/ops/attention/test_topk_v2.py: 新增 test_topk_v2_split_duplicate_scores(阈值 bin 重 tie)
```

```cpp
+      if (const auto [split, split_ws] = split_plan(batch_size, max_seq_len, device); split >= kSplitMin) {
+        LaunchKernel({batch_size, split}, kBlockSize, device).launch(topk_split_hist, params, split_ws);
+        LaunchKernel({batch_size, split}, kBlockSize, device).launch(topk_split_select<kMode>, params, split_ws);
+        return;
+      }
```

## 性能声明

MI355X、k=2048、page_size=64,对比父 commit 的 topk_v2:

- 71 个被 gate 命中的点全部 ≥1.00x:最差 1.06x、最佳 3.03x,geomean 1.56x。示例:B1/L128K 33.50→16.70 us(2.01x);B4/L256K 53.71→19.07 us(2.82x);B64/L128K 1.06x;B128(门外)1.00x。kernel 加速在 batch 1–8、128K+ 上下文为 1.8–3.0x。
- 端到端 GLM-5.2-MXFP4,TP4:batch 1、120K 上下文 decode top-k kernel 快 2.07x,median ITL 10.29→9.90 ms(-3.8%);600K 上下文 kernel 快 6.42x,ITL 12.32→10.02 ms(-18.7%)。无任何配置变慢,TTFT 不变。
- Agentic replay(vs packed-only topk v2 baseline):P90 interactivity TP8 +11.3%、TP4 +3.5%;每芯片 token 吞吐 TP8 +1.9%、TP4 +5.9%(geomean:c1/c4 与 c4/c8/c12)。
- 精度:GSM8K 1200 题 TP4 0.933→0.936、TP8 0.934→0.933,噪声内。

## 验证与回归风险

- `test_topk_v2.py` 在 gfx950 上 294 通过(288 既有 + 6 新);输出与 `torch.topk` 逐值一致(含 tie);新增重 tie 测试覆盖跨块 tie 合并的正确性边界。12-bit 直方图与被替换的 `TopKStreaming` 路径同宽。
- 回归风险:仅 ROCm decode 路径;CUDA 需 CI 确认真实 nvcc 编译(checklist 注明预处理 TU 已逐字节相同);`native topk.hip`(#37591,SGLANG_OPT_USE_TOPK_V2=0)未动。scratch 永不释放是 HIP graph 约束。CI 显示 Extra 与 AMD ROCm 10 失败,Base 通过。

## 关联

- 前序:#36851(topk_v2 成为 ROCm 默认)、#37591(native topk.hip 协作 kernel,另一实现)。
- 同区域:#40708(MoE 路由 topk,AMD)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
