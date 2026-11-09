# PR #1941: [Performance, Triton Kernel Args] extend_attention, optimize kern args to _fwd_kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HaiShaw |
| 合并 | 2024-11-08 |
| 分类 | Attention / Kernel(tier: major) |
| Label | 无 |
| Diff | pack diffstat 缺失;diff 为 extend_attention.py 约 10 行小改动 |
| 链接 | https://github.com/sgl-project/sglang/pull/1941 |

## 一句话结论

为 ROCm 上 extend_attention 的 Triton kernel 传入 waves_per_eu 等 HIP 调优参数,prefill 吞吐提升 2.2–2.8%。

## 动机

PR 正文极简:仅说明目的是"为 ROCm 上 extend_attention 的 `_fwd_kernel` 设置最优 kernel arguments",并给出两个模型的 prefill 吞吐加速比。正文未展开改动前的瓶颈分析或机制解释。

## 优化点

在 `extend_attention_fwd` 启动 Triton `_fwd_kernel` 时,HIP 路径额外传入三个 AMD 后端专有的 kernel 调优参数:`waves_per_eu=4`、`matrix_instr_nonkdim=16`、`kpack=2`;CUDA/非 HIP 路径传空 dict,launch 参数不变。既有 `num_warps`(Lk ≤ 64 时 4,否则 8)与 `num_stages=1` 保持原值。改动是纯 launch 配置,不触及 kernel 源码与调度逻辑。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py: HIP 路径为 _fwd_kernel 追加 extra_kargs
```

```python
extra_kargs = {}
if is_hip():
    extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

_fwd_kernel[grid](
    ...,
    num_warps=num_warps,
    num_stages=num_stages,
    **extra_kargs,
)
```

## 性能声明

- Llama-3.1-8B,TP=8,FP8,b32/i1024:prefill throughput +2.82%
- Grok-1,TP=8,FP8,b32/i1024:prefill throughput +2.23%

改动针对 ROCm/HIP,但 PR 未注明测试用的具体硬件型号与 workload 细节(batch/输入长度记号 b32/i1024 之外无更多信息)。

## 验证与回归风险

PR 未提供 accuracy 测试或新增回归测试的描述。改动只影响 HIP 路径的 kernel launch 参数,CUDA 路径 `extra_kargs` 为空 dict 不受影响。风险点:三个调优参数对不同 head dim、seq len 与 MIOPEN/Triton 版本的通用性未在正文论证,其他 HIP 模型路径继承同一参数组合。

## 关联

无(pack 未给出同区域其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
