# PR #2610: Change extend attention kernel launch parameter for ROCm platform to improve performance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @kkHuang-amd |
| 合并 | 2024-12-27 |
| 分类 | Attention / Kernel(major) |
| Label | 无 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示仅改动 extend_attention.py 的 launch 参数选择逻辑 |
| 链接 | https://github.com/sgl-project/sglang/pull/2610 |

## 一句话结论

ROCm 平台上把 Triton extend attention 的 launch 参数固定为 (BLOCK_M=64, BLOCK_N=64, num_warps=4),E2E 延迟 geomean 提升 1–2%。

## 动机

ROCm 平台上 `extend_attention_fwd` 此前复用 CUDA capability 驱动的 BLOCK_M/BLOCK_N 选择逻辑与按 `Lk` 决定的 num_warps,这些 tile 配置不是为 AMD GPU 调优的,在 ROCm 上存在可测的 E2E 延迟开销。PR 正文仅写 "As it is",未展开更多背景。

## 优化点

在 `extend_attention_fwd` 中为 HIP 平台单独分支:ROCm(`is_hip_`)下固定 `BLOCK_M, BLOCK_N = (64, 64)`、`num_warps = 4`;CUDA 与其他平台保持原有逻辑(CUDA capability >= 9 / >= 8 / 其余三档,`num_warps = 4 if Lk <= 64 else 8`,从 grid 计算前移到平台分支内)。改动是纯 launch 参数选择,不触及 kernel 本体。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py: extend_attention_fwd
  为 HIP 平台新增 launch 参数分支(64,64 + num_warps 4),CUDA 逻辑不变
```

关键摘录(diff):

```python
if is_hip_:
    BLOCK_M, BLOCK_N = (64, 64)
    num_warps = 4
else:
    if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
        ...
    num_warps = 4 if Lk <= 64 else 8
```

## 性能声明

PR 正文:改变 ROCm 平台 launch 参数后,median E2E latency 的 Geomean 有 1–2% 改善。未注明测试条件(模型、硬件型号、workload、batch 均未给出)。

## 验证与回归风险

PR 正文无精度/正确性测试说明,未新增测试;Motivation/Modifications 章节为空,验证信息不足。回归风险:CUDA 与其他平台路径逐分支保持原逻辑;ROCm 上所有走 Triton extend attention 的模型 tile 形状从随 Lq/Lk 变化改为固定 (64, 64)、num_warps 固定 4,不同序列长度/头维度下的表现未在正文分档说明。

## 关联

无(pack 内未提及关联 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
