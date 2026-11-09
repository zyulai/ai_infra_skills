# PR #8770: [1/2][resubmit] sgl-kernel: Fuse routed scaling factor into moe_fused_gate (select_experts)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trevor-m |
| 合并 | 2025-08-09 |
| 分类 | Attention / Kernel(tier: major) |
| Label | 无(素材标注 none) |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 topk.py、sgl-kernel CUDA kernel/绑定与单测 |
| 链接 | https://github.com/sgl-project/sglang/pull/8770 |

## 一句话结论

把 routed scaling factor 乘法融合进 moe_fused_gate CUDA kernel,省一次单独的 topk_weights 缩放步骤,BS1 prefill 快 10.46%。

## 动机

DeepSeek 类 MoE 的 `routed_scaling_factor` 此前在 select_experts 之后由单独的输出缩放步骤应用;`moe_fused_gate`(分层二段 topk 选择,含 shared experts 处理)已在 kernel 内完成归一化,顺势把缩放乘进 kernel 可省去一次独立的 elementwise 操作与中间张量。本 PR 是 #8364 的重新提交(上次遗漏了单测所需改动);PR 正文注明 #8690 将为 deepseek 启用该 fusion。

## 优化点

`moe_fused_gate` kernel(fixed 与 dynamic 两个变体)新增 `apply_routed_scaling_factor_on_output` bool 参数:在归一化 `output / output_sum` 的同一循环内按需乘上 `routed_scaling_factor`,无额外 pass。Python 侧 `TopKConfig` 新增同名配置并沿 `forward_cuda -> grouped_topk_gpu / biased_grouped_topk_impl` 透传;纯 PyTorch 路径(`grouped_topk_gpu`、`biased_grouped_topk_impl`)也在 renormalize 后直接 `topk_weights *= routed_scaling_factor`。`biased_grouped_topk_gpu` 把该参数传入 sgl_kernel 的 `moe_fused_gate`;aiter 路径、torch_native、fused_topk、custom_routing_function 等未实现融合的路径显式 assert 拒绝,避免静默错误。语义上 `num_fused_shared_experts > 0` 时 shared experts 权重此前已除以 scaling factor(设计为与后续输出缩放相消),开启本选项后缩放在 topk 阶段完成,注释同步更新。

## 关键改动

```text
sgl-kernel/csrc/moe/moe_fused_gate.cu: kernel 签名与 impl 增加 apply_routed_scaling_factor_on_output,归一化循环内缩放
sgl-kernel/python/sgl_kernel/moe.py: wrapper 新参数透传 torch.ops.sgl_kernel.moe_fused_gate
python/sglang/srt/layers/moe/topk.py: TopKConfig 新配置;grouped_topk_gpu / biased_grouped_topk_impl / biased_grouped_topk_gpu 透传或应用
sgl-kernel/tests/test_moe_fused_gate.py: 参数化 apply_routed_scaling_factor_on_output True/False 对比参考实现
```

```cuda
for (int ii = 0; ii < topk; ++ii) {
  int64_t const idx = topk * thread_row + ii;
  output_ptr[idx] = output_ptr[idx] / output_sum;
  if (apply_routed_scaling_factor_on_output) {
    output_ptr[idx] *= routed_scaling_factor;
  }
}
```

## 性能声明

PR 正文仅给出四组数字,未注明模型、硬件、精度与 workload 等测试条件:

- Prefill:BS 1 加速 10.46%,BS 128 加速 1.86%
- Decode:BS 1 加速 1.22%,BS 128 加速 0.26%

## 验证与回归风险

`sgl-kernel/tests/test_moe_fused_gate.py` 以 `biased_grouped_topk` 参考实现对拍,新增 `apply_routed_scaling_factor_on_output` True/False 参数化(覆盖 num_fused_shared_experts 0/1/2 与多组 num_experts/group/topk 组合)。风险:sgl-kernel 的 op schema 变更(新 bool 参数)要求 sgl-kernel 包同步升级,Python 侧留有 `assert not apply_routed_scaling_factor_on_output` 的 TODO 待 sgl-kernel 更新后解除;aiter 与 torch_native/custom routing 路径不支持该选项,误配置会显式报错;缩放从 MoE 输出侧挪到 topk 侧改变数值路径,依赖 `apply_routed_scaling_factor_on_output` 配置正确路由。本 PR 标题标注 [1/2],为系列两部分之一。

## 关联

前序:#8364(被退回的首次提交)、#8690(为 deepseek 启用该 fusion,后续)。其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
