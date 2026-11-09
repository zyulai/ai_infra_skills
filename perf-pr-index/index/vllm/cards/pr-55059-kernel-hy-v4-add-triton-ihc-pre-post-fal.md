# PR #55059: [Kernel][HY V4] Add Triton iHC pre/post fallback

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @linitra24 |
| 合并 | 2026-09-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, verified |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含新增 `vllm/models/hy_v4/nvidia/triton_ihc.py`、kernel benchmark 脚本,及 `hc.py` 调度修改 |
| 链接 | https://github.com/vllm-project/vllm/pull/55059 |

## 一句话结论

为 HY V4 的 iHC pre/post 增加树内 Triton 实现,八卡 H20 e2e 吞吐 +24.74%,kernel 级最高 12.83x。

## 动机

HY V4 的 iHC 三步(pre/post/head)此前只有两条路径:HPC-Ops 单 kernel 替换(需安装 hpc 包、`VLLM_ENABLE_HPC_OPS=1` 且仅支持 SM100/SM103)和通用 eager PyTorch 兜底。H20 是 SM90,只能走 eager,大量用户在 NVIDIA GPU 上没有可用的加速路径。需要一棵树内、可移植的 Triton 实现填中间层,同时保留 eager 兜底。Triton 实现改编自 sglang#36805。

## 优化点

新增 `triton_ihc_pre`/`triton_ihc_post` Triton kernel。pre 拆成两 stage:stage1 按 token × BLOCK_K split 并行,计算局部 sum of squares 与 pre/post mix 的部分和,写入 partial 缓冲;stage2 归并 partial 得 RMS 倒数、sigmoid 门,按 BLOCK_D tile 融合输出并仅在第一个 hidden block 写 post 门。post kernel 把子块输出按 channel gate 散射回 iHC residual 通道。调度为三级:启用且支持时用 HPC-Ops;否则 CUDA + FP16/BF16 时用 Triton(`triton_ihc_supported` 检查,HPC 之外还排除 `VLLM_BATCH_INVARIANT`);不支持时回落 eager。kernel 用 `enable_fp_fusion=False` 保证数值可控,pre 输出/门与 eager 对比容差分别为 2e-2/1e-5 与 1e-5,post 为逐位一致。

## 关键改动

```text
vllm/models/hy_v4/nvidia/triton_ihc.py: 新增文件,_ihc_pre_stage1/stage2 与
  _ihc_post_kernel Triton kernel 及 triton_ihc_supported 门控
vllm/models/hy_v4/nvidia/hc.py: HYV4HCPreLayer/HYV4HCPostLayer forward
  在 hpc_op 之后插入 Triton 快路径
benchmarks/kernels/benchmark_hy_v4_ihc.py: 新增 CUPTI 冷 L2 计时基准
```

关键片段(调度顺序):

```python
if self.hpc_op is not None:
    return self.hpc_op(x)
if triton_ihc_supported(x):
    return triton_ihc_pre(x, self.hc_fn.weight, self.hc_scale, ...)
# eager fallback
```

## 性能声明

Kernel 基准(单卡 H20 SM90,CUPTI 计时、冷 L2,FP16/BF16,hidden 4096/6144,1-8192 tokens):iHC pre 快 2.88x-9.97x,iHC post 快 3.91x-12.83x(vs eager)。E2E:Hy4-preview-FP8,8×H20,TP=8,MTP 与 prefix caching 关闭,100 请求、2048 输入/256 输出、10 req/s:输出吞吐 214.51 → 267.57 tok/s(+24.74%),总吞吐 1930.55 → 2408.13 tok/s(+24.74%),Mean TTFT 42634.72 → 32370.70 ms(-24.07%),Mean TPOT 275.70 → 223.66 ms(-18.87%)。精度:GSM8K 全集(lm-eval,5-shot,1319 样本)Eager 86.58%/85.60% vs Triton 86.20%/85.97%,差异 0.38pp 小于约 0.94-0.97pp 标准误。HPC-Ops 未参与对比(仅支持 SM100/SM103)。

## 验证与回归风险

验证:kernel 基准内嵌 `torch.testing.assert_close` 数值对比;GSM8K 精度回归;e2e serving 对比。风险:Triton 路径要求 weight/scale/base 为 FP32 contiguous,x 需 contiguous,不满足会 assert;pre 数值容差较大(atol 2e-2),与 eager 存在轻微数值差异;`VLLM_BATCH_INVARIANT` 下仍走 eager 以保持 batch 不变语义。head 步(`HpcIHCHead`)及跨层 post+pre 融合(`HpcIHCPostPre`)未移植,后者因需重构 decoder layer 调度列为 TODO。

## 关联

改编自 sgl-project/sglang#36805。同分类 Kernel/Attention 类 PR:#60084、#42235、#42191、#27284 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
