# PR #23809: [fix]: add Arm 4bit fused moe support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nikhil-arm |
| 合并 | 2025-09-24 |
| 分类 | MoE |
| Label | ready, ci/build |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/23809 |

## 一句话结论

为 Arm(AArch64)CPU 新增 fused int4 MoE 内核与 W4A8 量化方法,DeepSeek-V2-Lite 最高 3x。

## 动机

vLLM 的 CPU 后端此前没有 fused 4-bit MoE 路径:compressed-tensors 量化 MoE 模型在 Arm 上只能走非 fused 的通用实现,权重反量化与 expert 计算分散。Arm 平台 PyTorch 已提供基于 KleidiAI 的 `_dyn_quant_matmul_4bit`/`_dyn_quant_pack_4bit_weight` ATen 算子(仅 AArch64 可用),可以把 int4 权重 matmul、激活量化和 expert 分发融合到一条 CPU 路径上。

## 优化点

新增 C++ 内核 `dynamic_4bit_int_moe_cpu`:先按 topk_ids 把 token 分桶到 expert(计数 + 前缀和偏移),再用 `at::parallel_for` 按 expert 并行执行——`index_select` 取出本 expert 的 token,经 W13 packed 权重 4-bit matmul、激活(SiLU-and-mul / SwiGLU-Gu / GPT-OSS 风格 SwiGLUOAI 带 clamp,按 `activation_kind` 分派)、W2 matmul,router 权重按 `apply_router_weight_on_input` 在输入端或输出端相乘,最后 `index_add` 写回输出。权重在 `process_weights_after_loading` 中离线打包:`int4(存为 int8 [-8,7])` 加 8 移位成 nibble 对打包成 uint8,scales 按 channel-wise(float32)或 group-wise(bfloat16)与 bias 一起传入 `_dyn_quant_pack_4bit_weight`。Python 侧新增 `CompressedTensorsW4A8Int8MoEMethod`(CPU-only,校验动态 per-token 激活、4-bit 权重与 Arm 架构下 `_dyn_quant_*` 算子存在)。另将 CPU `select_experts` 默认 softmax-topk 改为先对 router logits 直接 topk 再对 top-k 值做 softmax(renormalize)或用 logsumexp 修正(不 renormalize),避免对全部 expert 先做完整 softmax。

## 关键改动

```text
csrc/moe/dynamic_4bit_int_moe_cpu.cpp: 新增 fused 4-bit MoE CPU 内核(分桶+按 expert 并行)
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py: 新增 CompressedTensorsW4A8Int8MoEMethod
vllm/model_executor/layers/fused_moe/cpu_fused_moe.py: select_experts 改为 topk-on-logits 再 softmax
csrc/cpu/torch_bindings.cpp / cmake/cpu_extension.cmake: 注册 op 并加入编译
```

```python
topk_logit_vals, topk_idx = torch.topk(router_logits, k=top_k,
                                       dim=-1, sorted=False)
if renormalize:
    topk_vals = torch.softmax(topk_logit_vals, dim=-1)
else:
    logZ = torch.logsumexp(router_logits, dim=-1, keepdim=True)
    topk_vals = (topk_logit_vals - logZ).exp()
```

## 性能声明

PR 正文仅一句:"Tested with Deepseek v2 lite model. The generation is correct with upto 3x speedups."——即 DeepSeek-V2-Lite 上生成正确、最高 3x 加速。未注明测试硬件(CPU 型号)、线程数、batch/token 规模与对照基线(推测为原非 fused 路径,PR 未写明)。无逐项 benchmark 数据。

## 验证与回归风险

PR 的 Test Plan 章节为空,未列 CI 测试;验证仅作者自测 DeepSeek-V2-Lite 生成正确性。风险面:内核仅 AArch64 可用(非 Arm 构建下 `TORCH_CHECK(false)`),但 `torch_bindings.cpp` 的注册无架构 guard;`CompressedTensorsW4A8Int8MoEMethod` 不支持 expert_map/EP 与 EPLB(apply 中 assert);`layer.py` 删除 CPU 平台的 `pass` 分支,使 CPU 走统一 `grouped_topk` 导入;`select_experts` 改写影响 CPU 平台所有默认 softmax 路由的 MoE 模型。

## 关联

无(pack 素材中未列出同区域前序/后续 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
