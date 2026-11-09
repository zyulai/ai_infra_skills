# PR #1845: [Performance, Triton Kernel Args] _decode_grouped_softmax_reducev_fwd speedup on ROCm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HaiShaw |
| 合并 | 2024-10-31 |
| 分类 | Attention / Kernel (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/1845 |

## 一句话结论

为 decode attention stage-2 Triton kernel 设置 ROCm 最优 launch 参数,MI300x 上 decode 吞吐中位数 +1.0%。

## 动机

Triton 在 AMD 后端上默认的 launch 配置并不针对 MI300 系列优化:AMD 的 ROCm 微调文档与 Triton AMD compiler 都支持通过额外 kernel 参数控制 waves per EU、WMMA 指令维度与 k-pack。SGLang 的 decode attention 第二阶段 kernel `_fwd_grouped_kernel_stage2`(`_decode_grouped_softmax_reducev_fwd`,softmax 后对 V 加权归约)此前用默认参数 launch,在 MI300x 上留有调优空间。本 PR 只调 launch 参数,不改 kernel 体。

## 优化点

在 `_decode_grouped_softmax_reducev_fwd` 里检测 `is_hip()`,是 ROCm 则给 `_fwd_grouped_kernel_stage2` 传入 `extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}`(来源注释指向 ROCm 6.2 的 Triton kernel 优化文档与 Triton third_party/amd compiler.py):`waves_per_eu=4` 提高 EU 占用,`matrix_instr_nonkdim=16` 对齐 WMMA 非该 K 维度,`kpack=2` 增大 K 维打包。CUDA 路径 `extra_kargs` 为空 dict,launch 行为不变。kernel 本体、grid、num_warps/num_stages 均未改动。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:
  _decode_grouped_softmax_reducev_fwd 增加 HIP 专属 extra_kargs 并透传给
  _fwd_grouped_kernel_stage2 launch
```

```python
extra_kargs = {}
if is_hip():
    # ROCm 6.2 optimizing-triton-kernel 文档 / triton amd backend compiler
    extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
_fwd_grouped_kernel_stage2[grid](..., num_warps=num_warps, num_stages=1, **extra_kargs)
```

## 性能声明

PR 正文:"~1.0% improvement to median decode throughput on MI300x with Grok-1 and FP8 (b32/i1024/o256)"。即 MI300x、Grok-1、FP8、batch 32 / input 1024 / output 256 的测试设置下 decode 吞吐中位数约 1.0% 提升;无 kernel 级单独计时数据,无其他硬件/模型数字。

## 验证与回归风险

PR 正文无 accuracy 测试小节,diff 不含测试改动;数值路径未变(仅 launch 参数),风险集中在性能而非正确性。回归风险:`waves_per_eu=4`/`kpack=2` 对其他 HIP 卡(MI200/MI300 以外)或极端 shape 可能不是最优甚至更慢;`matrix_instr_nonkdim` 依赖 Triton AMD backend 支持的版本;非 HIP 路径不受影响。

## 关联

同作者同一序列的 #1980 对 stage-1 kernel(`_decode_grouped_att_m_fwd`)与 fused_moe_kernel 做了同类 ROCm 参数与 mask 优化(本 PR 之后一天合并),两者共同覆盖 decode attention 两个 stage;其余同区域编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
