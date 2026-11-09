# PR #19500: [Hardware][NVIDIA][kernel] Fp4 MOE quant kernel optimization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jiahanc |
| 合并 | 2025-06-14 |
| 分类 | MoE |
| Label | performance, ready |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/19500 |

## 一句话结论

重写 Blackwell 上 MoE 的 FP4 activation 量化内核(grid-strice + 分 expert 数/负载策略),大 batch 下耗时较原实现降约 35%。

## 动机

cutlass MoE 流程中的 fp16→FP4 experts 量化内核原实现是"每个 block 处理一行"的布局,并行度不足;每行还要线性扫描 256 个 expert 的 offset 找 expert index,大 M 下 offset 可被 block 复用却反复从全局内存读。M top-k 越大,该量化步骤在 MoE 路径中的占比越明显。

## 优化点

四项机制改动(均作用于 `nvfp4_experts_quant.cu`):其一,内核改为全局 grid-stride 循环,按 `globalIdx` 展开到 `numRows * colsPerRow` 个工作项,而非每 block 一行;其二,host 侧 launch 逻辑在 grid 尚未覆盖全部 SM 且 block 较大时(`grid.x <= multiProcessorCount && block.x > 64`)逐次倍增 grid、对半缩小 block,换取更好占用率;其三,按 blockRepeat 区分两种路径:block 会被复用的大问题(LARGE_M_TOPK)新内核把 expert offsets 经 int4 向量化装载进 shared memory,再做二分查找定位 expert;小问题(block 不复用)则在寄存器数组 `local_offsets[17]` 中按 16 个一组分段缓存 offsets 后再搜索,`SMALL_NUM_EXPERTS < 4` 时直接逐个读;其四,n_experts>=16 与 <16 分别实例化模板。

## 关键改动

```text
csrc/quantization/fp4/nvfp4_experts_quant.cu: cvt_fp16_to_fp4 内核重写(grid-stride、shared/register offsets、二分查找)+ quant_impl launch 策略
```

```python
while (grid.x <= multiProcessorCount && block.x > 64) {
  grid.x *= 2;
  block.x = (block.x + 1) / 2;
}
```

## 性能声明

NVIDIA B200,nvidia/DeepSeek-R1-FP4,num_experts=256,topk=8(表值 PR 未注明单位,应为耗时,越低越好;基准为 triton MoE):

MKN=(4,2048,7168):cutlass fp4 w/o 9.5 → w/ 7.4;CUDA graphs 下 8.3 → 6.2(triton_moe 为 9.6 / 8.9)。MKN=(8,..):13.0→12.6。MKN=(16,..):19.2→18.0。MKN=(32,..):27.2→27。MKN=(64,..):36.9→35.4。MKN=(128,..):41.3→39.2。MKN=(256,..):42.8→39.5。MKN=(512,..):50.9→42.9。MKN=(1024,..):62.4→47.1。MKN=(2048,..):87.7→57.5(cuda graphs 86.3→56.0;triton_moe 126.6 / 125.9)。即大 M 下相对本内核原实现约 -35%,且明显快于 triton 路径。基准脚本 `benchmarks/kernels/benchmark_cutlass_fp4_moe.py`。

## 验证与回归风险

PR 测试计划仅给出性能基准入口,未列单测;diff 未含测试文件改动。风险面限于 SM100(Blackwell,`__CUDA_ARCH__ >= 1000` 分支)的 FP4 MoE 量化内核;量化精度路径未变(仅索引/调度策略变化),但 256 expert 假设下的 `local_offsets[17]` 分段搜索逻辑对 expert 数 <16/≥16 的分支划分需依赖现有 CI 覆盖。

## 关联

同分类(MoE)kernel 优化:PR #29354(fused MoE unpermute-aware 小 batch 路径)。Blackwell FP4 相关。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
