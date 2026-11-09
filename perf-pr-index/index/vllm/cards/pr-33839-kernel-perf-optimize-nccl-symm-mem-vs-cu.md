# PR #33839: [Kernel][perf] optimize NCCL symm_mem vs custom_AR selection thresholds

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pkousha |
| 合并 | 2026-02-26 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/33839 |

## 一句话结论

改 allreduce 选择为区间判定，小 tensor 也走 NCCL symm_mem，8xH100 最高 6.14x。

## 动机

正文指出旧的阈值逻辑只在大张量上启用 NCCL symmetric memory allreduce（4 GPUs ≥2MB、8 GPUs ≥1MB）。但 H100 / GB200 benchmark 显示，symm_mem 在小张量（≤16KB）上也显著优于 custom_AR（1.21x-1.48x speedup），旧逻辑把这些场景的性能留在了桌上。

## 优化点

把单一 size threshold 换成 range-based 判定：显式定义 custom_AR 占优的中间区间，在区间之外（小与大张量）都改用 symm_mem。配置 `custom_ar_preferred_ranges` 为 4 GPUs `(16 * KiB, 512 * KiB)`、8 GPUs `(16 * KiB, 128 * KiB)`；>8 GPUs 保持 always symm_mem 不变。这样 4 GPUs 的 symm_mem 适用范围从"仅 ≥2MB"变为"≤16K 且 ≥512K"，8 GPUs 从"仅 ≥1MB"变为"≤16K 且 ≥128K"。

## 关键改动

```text
vllm/distributed/device_communicators/all_reduce_utils.py: 以 custom_ar_preferred_ranges 区间判定替换原单一 size threshold，决定使用 NCCL symm_mem 还是 custom_AR
```

核心配置（摘自 PR 正文 Solution 段）：

```python
"custom_ar_preferred_ranges": {
    4: (16 * KiB, 512 * KiB),  # custom_AR wins for 32K-256K
    8: (16 * KiB, 128 * KiB),  # custom_AR wins for 32K-64K
}
```

## 性能声明

测试条件：使用 `benchmarks/kernels/benchmark_device_communicators.py`，8x H100 与 4x H100 单节点 NVLink，tensor sizes 2KB 到 1GB，每配置 100 trials、启用 CUDA graph，环境 `NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_USE_NCCL_SYMM_MEM=1`；精度 PR 未注明。8 GPUs H100：2K-16K 选 PyNCCL-symm，比 custom AR-1stage 快 1.35x-1.48x；32K-64K custom_AR-1stage 为 baseline；128K-1GB PyNCCL-symm 快 1.12x-6.14x。4 GPUs H100：2K-16K PyNCCL-symm 比 custom_AR 快 1.21x-1.30x；32K-256K custom_AR-1stage 比 PyNCCL-symm 快 1.07x-1.35x；512K-1GB PyNCCL-symm 比 custom_AR 快 1.10x-2.32x。细粒度延迟（8 GPUs）：2 KB custom_AR 6.8 μs vs symm 4.6 μs（1.48x）、16 KB 7.2 vs 5.3 μs（1.35x）、1 MB 58.3 vs 32.1 μs（1.82x）、16 MB 812.4 vs 132.3 μs（6.14x）。

## 验证与回归风险

正文给出 benchmark 命令（`torchrun --nproc_per_node=8 ... --num-trials 100 --use-cuda-graph`）与上述结果表，并附 H100 单节点、B200 单节点的性能图（B200 未给具体数字）。PR 未注明引入的单元测试。回归风险：>8 GPUs 行为保持 unchanged；正文未说明对非 NVLink 拓扑或其他硬件的回归评估。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
