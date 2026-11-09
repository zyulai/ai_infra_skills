# PR #12996: Support piecewise cuda graph for deepseek v3

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-11-10 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/12996 |

## 一句话结论

为 DeepSeek V3 打通 piecewise CUDA graph,bs1 时 mean TTFT 降低 43%。

## 动机

本 PR 是 #11812 的后续,目标是让 DeepSeek-R1(DeepseekV3ForCausalLM)在 prefill 阶段用上 piecewise CUDA graph(`--enable-piecewise-cuda-graph --piecewise-cuda-graph-max-tokens 8192`)。prefill 此前无法像 decode 一样整体用 CUDA graph 捕获;而 torch.compile 要求每个自定义算子都有 fake 实现,MoE 路由的 `sgl_kernel::moe_fused_gate` 缺少注册,图会在该算子处断开,因此需要补齐算子注册与显存预留等配套。

## 优化点

在 `topk.py` 用 `torch.library.register_fake` 为 `sgl_kernel::moe_fused_gate` 注册 fake 实现,声明输出为 `(num_rows, topk)` 的 fp32 topk_weights 与 int32 topk_ids,使 MoE 路由可被 torch.compile 追踪,piecewise CUDA graph 得以分段捕获 prefill。`server_args.py` 在启用该特性时按 `piecewise_cuda_graph_max_tokens // 4` 追加 reserved_mem,避免 capture 挤占 KV cache 空间。对 DeepseekV3ForCausalLM 且非 NSA 的配置,启用时打印 "Piecewise CUDA graph is enabled, use MLA for prefill.",即 prefill 走 MLA 路径。

## 关键改动

```text
python/sglang/srt/layers/moe/topk.py: 为 sgl_kernel::moe_fused_gate 注册 fake 实现,修复 torch.compile 断图
python/sglang/srt/server_args.py: piecewise CUDA graph 显存预留与 DeepSeek V3 prefill-MLA 提示
```

## 性能声明

DeepSeek-R1-0528、TP=8、`--piecewise-cuda-graph-max-tokens 8192`,bench_serving random 数据集(input 1024 / output 10,random-range-ratio 0.98),开/关 piecewise CUDA graph 对比:mean TTFT 69.214ms vs 122.417ms(cc=1)、162.667 vs 227.405(cc=4)、455.605 vs 797.197(cc=16)、770.717 vs 891.407(cc=32);正文结论 "Mean TTFT is reduced **43%** for bs1"。input throughput cc=1 为 6525.850 vs 4847.049 tok/s。硬件型号 PR 未注明。

## 验证与回归风险

PR 正文未列新增单测。风险点:fake 实现的输出 dtype/shape 需与真实 kernel 一致,影响所有走 torch.compile / piecewise CUDA graph 的 DeepSeek V3 类模型;`mem_fraction_static` 计算新增预留项,改变默认显存分配;`_handle_model_specific_adjustments` 中 DeepSeek V3 默认 backend 选择逻辑被触碰。

## 关联

前序 PR #11812(piecewise CUDA graph 基础支持)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
