# PR #14640: [sgl-kernel][Feat][B200][2/N] Support MXFP8 Grouped GEMM in Blackwell

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HydraQYH |
| 合并 | 2026-02-26 |
| 分类 | MoE(major) |
| Label | high priority, quant, sgl-kernel, run-ci |
| Diff | 素材未注明 diffstat(diff 片段可见 1 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/14640 |

## 一句话结论

将 B200 上 MXFP8 分组量化内核的 group-wise 调度改为 flatten 调度,E2E 显存带宽利用率从 75% 提升到 81%。

## 动机

承接 #13731 的 MXFP8 Grouped GEMM 工作。此前的 MXFP8 Group Quant 内核采用 group-wise 调度:每个 group 内 CTA 编号从 0 重新映射 tile。当各 group 的 tile 数不均衡时,这种调度会造成负载不均(部分 CTA 在 group 边界处闲置),因此改为 flatten 调度让 CTA 编号跨 group 连续。

## 优化点

`mxfp8_group_quant` 内核把 per-group 独立分配 CTA 的写法改为全局连续编号:维护跨 group 累计的 `group_total_tiles` 与 `head_cta_id`,每个 CTA 的 tile 偏移按 `(blockIdx.x + (gridDim.x - head_cta_id)) % gridDim.x` 计算,使 CTA→tile 的映射跨 group 连续并循环回绕(Group X 的尾部 tile 可以由 CTA 0 接着处理)。这样无论各 group 的 M/K 尺寸如何不均,所有 CTA 都能持续拿到工作,消除 group 边界处的空转;内核其余量化逻辑不变。

## 关键改动

```text
sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh: mxfp8_group_quant 内核调度从 group-wise 改为 flatten
```

```cuda
-    decltype(total_tiles) blk_offset = blockIdx.x;
+    group_total_tiles += total_tiles;
+    auto blk_offset = (blockIdx.x + (gridDim.x - head_cta_id)) % gridDim.x;
+    head_cta_id = group_total_tiles % gridDim.x;
```

## 性能声明

沿用 #13731 中的同一算例验证性能:改用 flatten 调度后,E2E memory bandwidth utilization 从 75% 提升到 81%(PR 正文以 Nsight 截图形式给出 before/after,未标注具体模型、batch 与精度条件;标题与内核文件表明平台为 Blackwell B200/sm100)。PR 正文 Accuracy Tests 一节为空,未提供精度数据。

## 验证与回归风险

PR 未新增单元测试(checklist 中 Add unit tests 未勾选)。改动集中在 sm100 expert specialization 的 MXFP8 blockscaled group 量化内核,回绕式 CTA→tile 映射的正确性是主要风险点:若 `head_cta_id` 累计有误,可能出现 tile 重复量化或遗漏;影响路径为 Blackwell 上使用该 MXFP8 分组量化的 MoE 专家流水线。CUDA/ROCm 以外的平台不受影响(该 .cuh 为 sm100 专用)。

## 关联

前序:PR #13731(MXFP8 Grouped GEMM 在 Blackwell 的引入,[B200] 系列第 1 步,本 PR 为 2/N)。同分类:Blackwell MoE/量化内核相关 PR(素材未列出其他编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
