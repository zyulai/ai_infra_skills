# PR #8364: [1/2] sgl-kernel: Fuse routed scaling factor into select_experts

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trevor-m |
| 合并 | 2025-08-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/8364 |

## 一句话结论
把 `routed_scaling_factor` 乘法融合进 `moe_fused_gate` select_experts kernel,DeepSeek-R1 FP4 prefill BS1 提速 10.46%。

## 动机
#8333 的后续:非 FP4 路径已把 `routed_scaling_factor` 融合进 `moe_sum_reduce`,但走 `moe_fused_gate` 的路径仍在 select_experts 之后单独做这次乘法。PR 参照 TRT-LLM DeepSeek-V3 实现(`noAuxTcKernels.cu`)把该缩放直接并入 gate kernel 输出,省去一次独立的逐元素乘。

## 优化点
给 `moe_fused_gate` CUDA kernel 增加 `bool apply_routed_scaling_factor_on_output` 参数:在 topk 权重归一化(`output_ptr[idx] / output_sum`)后,若开关为真则在同一 kernel 内乘以 `routed_scaling_factor`。同步修改 static/dynamic 两个 kernel 变体、`common_extension.cc` 的 op schema、`sgl_kernel_ops.h` 声明与 Python 绑定 `sgl_kernel/moe.py`(默认 `apply_routed_scaling_factor_on_output=False` 保持旧行为);注释同时澄清 `num_fused_shared_experts > 0` 时 shared experts 会被除以 `routed_scaling_factor` 以便后续合并时抵消。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_fused_gate.cu: 归一化后按 flag 乘 routed_scaling_factor(impl/kernel/kernel_dynamic)
sgl-kernel/csrc/common_extension.cc / include/sgl_kernel_ops.h / python/sgl_kernel/moe.py: op 签名与 Python 绑定加参
sgl-kernel/tests/test_moe_fused_gate.py: 对 apply_routed_scaling_factor_on_output True/False 双向参数化
```

## 性能声明
`nvidia/DeepSeek-R1-0528-FP4`、`--quantization modelopt_fp4 --tp 8 --enable-flashinfer-cutlass-moe --enable-ep-moe`、input/output len 1024(`bench_one_batch_server`):Prefill BS 1 +10.46%(input throughput 9392.46→10375.70 tok/s)、BS 128 +1.86%(17344.15→17667.47 tok/s);Decode BS 1 +1.22%、BS 128 +0.26%(output throughput 4066.24→4076.75 tok/s)。GSM8K 1319 题 8-shot:0.960→0.961。

## 验证与回归风险
`test_moe_fused_gate.py` 在既有 seq_length × expert 配置 × num_fused_shared_experts 参数化之上,对 `apply_routed_scaling_factor_on_output` True/False 增加 参数化,与 `biased_grouped_topk` 参考实现对拍。风险:开关默认 False,调用方需显式传 True 才有行为变化;语义上 shared experts 在融合路径中被除以 scaling factor 以抵消后续缩放,与调用侧的合并逻辑强耦合(此为 [1/2] 系列,PR 标题表明还有后续 PR)。

## 关联
前序 #8333;后续为 [2/2](PR 编号素材未给出)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
