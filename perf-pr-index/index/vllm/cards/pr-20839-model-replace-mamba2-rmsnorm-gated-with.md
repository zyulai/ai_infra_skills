# PR #20839: [Model] Replace Mamba2 RMSNorm Gated with Fused Triton Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @cyang49 |
| 合并 | 2025-07-25 |
| 分类 | Attention / Kernel(major) |
| Label | ready |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/20839 |

## 一句话结论

用 mamba_ssm 的融合 Triton kernel 替换 Mamba2 RMSNormGated 的 unfused 实现,Bamba-9B 长前缀延迟降约 8–9%。

## 动机

原 RMSNormGated CUDA 实现是 unfused 操作:先对 gate 做 silu、与 x 相乘,再调用 `ops.rms_norm`,中间产生多份中间张量。实验发现 16k/32k prompt 长度下,RMSNormGated 层耗时与 Mamba2 SSD 计算相当,远超预期,是长上下文 prefill 的明显瓶颈。

## 优化点

从 state-spaces/mamba 仓库移植 `layernorm_gated.py` 的单 pass Triton kernel:`_layer_norm_fwd_1pass_kernel` 在一个 kernel 内完成 RMSNorm 统计(`x*rstd`)与 gated 激活(`z * sigmoid(z)` 乘法),支持 `NORM_BEFORE_GATE` / `IS_RMS_NORM` 等 constexpr 分支,按 group 划分 grid `(M, ngroups)`。Mamba2 mixer 的 `forward_cuda` 在 TP=1 且 ngroups=1 时改调 `rms_norm_gated(x, weight, z=gate, norm_before_gate=False)`,省去 silu-mul 的临时张量与第二次读写;TP>1 或 ngroups≠1 仍走 `forward_native`。

## 关键改动

```text
vllm/model_executor/layers/mamba/ops/layernorm_gated.py: 新增,移植 mamba_ssm 的 gated RMSNorm Triton kernel
vllm/model_executor/layers/mamba/mamba_mixer2.py: forward_cuda 在 TP=1/ngroups=1 时改用 rms_norm_gated
```

## 性能声明

H100-80GB,`ibm-ai-platform/Bamba-9B-v2`,`benchmark_latency.py` batch=1、output-len=1:16k 输入平均延迟 0.6323s → 0.5812s;32k 输入 1.2799s → 1.1791s;约 8~9% latency improvement(before 基线为 commit f29fd8a)。GSM8K 5-shot(lm_eval,TP=1):flexible-extract 0.4162 → 0.4246,strict-match 0.4132 → 0.4246。

## 验证与回归风险

测试:e2e `lm_eval` GSM8K 精度对齐(见上,略有提升)+ `benchmark_latency` 性能对比;pack 未列出新增单元测试。回归风险:作者明确说明当前只覆盖 TP=1 且 ngroups=1 的场景,更广用例("broader use cases")可能需要类似修复;kernel 对 feature dim ≥ 64KB 报 RuntimeError(沿用原实现限制);输出 dtype/精度路径由 Triton kernel 的 fp32 计算决定,与原 `ops.rms_norm` 链条可能有细微差异(PR 以 e2e 精度验证兜底)。

## 关联

同区域为 Mamba2 / mamba_ssm kernel 移植;pack 正文未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
