# PR #34933: [FIX] fused moe with lora shared expert dual stream (1.07x otps)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jhaotingc |
| 合并 | 2026-02-25 |
| 分类 | MoE(major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/34933 |

## 一句话结论

修复 LoRA 使 MoE shared expert 失去双 stream 重叠的问题,恢复 TP 下 shared/routed expert 并行,吞吐提升至约 1.07x。

## 动机

带 shared expert 的 MoE 模型在 TP 下原本有双 stream 重叠:shared expert 与 routed expert 在不同 CUDA stream 上并行。但一旦启用 LoRA 模块,该重叠消失——因为 LoRA 注入路径把 `shared_experts` 挂到了 MoE modular kernel 内部,kernel 整体在单 stream 上执行,shared expert 的并行机会被吞掉。

## 优化点

改动很小但关键:在 `FusedMoEWithLoRA._inject_lora_into_fused_moe` 中,当走 quant method 自带的 modular kernel(`supports_internal_mk`)时,显式 `m_fused_moe_fn.shared_experts = None`;走 `select_gemm_impl` 新建 `FusedMoEModularKernel` 的分支则不再把 `self.base_layer.shared_experts` 传入。即 kernel 不再拥有 shared experts,将其交还给 runner,由 runner 用独立 CUDA stream 把 shared expert 与 routed expert 重叠执行。LoRA 的注入逻辑本身不变,只是改变了 shared expert 的归属,使其回到与无 LoRA 时相同的双 stream 调度。

## 关键改动

```text
vllm/lora/layers/fused_moe.py: LoRA 注入时不再把 shared_experts 归入 MoE kernel,交还 runner 双 stream 重叠
```

关键片段:

```python
if getattr(self.base_layer.quant_method, "supports_internal_mk", False):
    m_fused_moe_fn = self.base_layer.quant_method.moe_mk
    # Don't let the kernel own shared experts so the runner can
    # overlap them with routed experts via a separate CUDA stream.
    m_fused_moe_fn.shared_experts = None
```

## 性能声明

vllm bench serve(random 数据集,input_len=1600、output_len=600、prefix=0,ignore-eos),模型 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` + LoRA(r16 zeros),结果目录名含 H200_TP1_1lora(单卡 H200、1 个 LoRA)。并发 1→64 的 OTPS/GPU speedup 依次为 1.08 / 1.07 / 1.06 / 1.07 / 1.05 / 1.05 / 1.02(如并发 1:128.89→138.67 OTPS/gpu;并发 32:1841.87→1937.97)。标题标称 1.07x otps。正确性:gsm8k 5-shot lm-eval(TP4 serve + enable-lora 无 LoRA 请求)before flexible-extract 0.4291 / after 0.4238,strict-match 0.6528 / 0.6505,作者判定无变化。

## 验证与回归风险

验证方式:lm-eval gsm8k(上文数字)+ nsys 确认启用 LoRA 时引擎确实走 LoRA 路径;serve 配置含 `--enable-lora --max-loras 1 --max-lora-rank 16` 与 cudagraph capture sizes {1..128}。风险面:改动位于 LoRA × shared-expert MoE 交叠处,影响所有启用 LoRA 且带 shared expert 的 MoE 模型的 kernel 组装;若 runner 侧双 stream 调度未接管 shared experts(如非 TP、其他 quant method),需确认没有路径遗漏执行 shared expert。PR 未注明新增自动化测试。

## 关联

同区域:PR #42290(MoE LoRA one-shot kernel,含 dual stream 接入 FusedMoEWithLoRA)、`VLLM_LORA_ENABLE_DUAL_STREAM` 相关改动;PR 未注明其他前序/后续编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
