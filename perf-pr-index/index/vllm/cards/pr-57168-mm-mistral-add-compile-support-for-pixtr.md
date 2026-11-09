# PR #57168: [MM][Mistral] Add compile support for Pixtral vision encoders

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jzakrzew |
| 合并 | 2026-09-30 |
| 分类 | Model-specific Fast Path（tier: major） |
| Label | ready, torch.compile, mistral |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files); pack 末尾无 dropped-from-diff 说明(diff 未裁剪文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/57168 |

## 一句话结论

Pixtral 视觉编码器接入 torch.compile，16 图 BF16 下 GPU kernel 时间降 42%（1.73×），E2E 提升更小。

## 动机

为两种 Pixtral 视觉编码器实现补齐 `compile_mm_encoder` 支持：`PixtralForConditionalGeneration`（含 Mistral-format 的 Mistral 3 模型）与 `Mistral3ForConditionalGeneration` 使用的 Hugging Face Pixtral encoder。此前这两类 encoder 的 transformer block 未纳入编译，elementwise、SiLU、Concat 等琐碎开销在执行中占比很大。正文特别注明结果只反映 GPU kernel 时间：即便配合 #56277 的 CUDA graphs，总 E2E 运行时提升也小得多，因为当前主瓶颈是 `apply_hf_processor`。

## 优化点

对两种实现的 transformer blocks 应用 `support_torch_compile`，并把 token、rotary embedding、packed-sequence 维度标记为 dynamic，使编译能适配可变形状；`position_embeddings` 的类型相应改为 `tuple[torch.Tensor, torch.Tensor]`。功能仍由 `compile_mm_encoder` 门控，且默认关闭。按正文给出的 kernel 分类对比，收益来自消除或融合执行开销：Elementwise 368 → 11 ms（-97%）、SiLU 121 → 44 ms（-64%）、Concat 96 → 0.3 ms（>99%），而 FlashAttention 273 → 273 ms、GEMM 272 → 272 ms 基本不变，说明编译主要削减的是算子启动与逐元素/拼接类开销而非矩阵乘本身。

## 关键改动

```text
vllm/model_executor/models/pixtral.py: 给 TransformerBlock 加 @support_torch_compile（dynamic_arg_dims 标记 x/freqs_cis/cu_seqlens/sequence_lengths，enable_if=should_torch_compile_mm_encoder，is_encoder=True），并把 position_embeddings 改为 tuple
tests/compile/fullgraph/test_multimodal_compile.py: 新增 test_pixtral_compilation，覆盖 pixtral 与 mistral3-hf 两种实现、编译开/关两档
```
```text
@support_torch_compile(
    dynamic_arg_dims={
        "x": 1,
        "freqs_cis": 0,
        "cu_seqlens": 0,
        "sequence_lengths": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class TransformerBlock(nn.Module):
```

## 性能声明

NVIDIA GB200 系统，用同一份 16-image BF16 capture 在启用编译前后对比：Total GPU kernel time 由 1173.6 ms 降至 676.8 ms，下降 42%，对应 1.73x speedup。分类明细：Elementwise 368 → 11 ms（-97%）、SiLU 121 → 44 ms（-64%）、Concat 96 → 0.3 ms（>99%）、FlashAttention 273 → 273 ms（0%）、GEMM 272 → 272 ms（0%）。测试条件：硬件 NVIDIA GB200、精度 BF16、workload 16-image capture、指标为 GPU kernel time；正文明确该结果仅关于 GPU kernel time，E2E 提升要小得多，并说明因 `apply_hf_processor` 成为主瓶颈。

## 验证与回归风险

测试命令 `.venv/bin/python -m pytest tests/compile/fullgraph/test_multimodal_compile.py -k pixtral_compilation -vv -s`，结果 4 passed、4 deselected；测试对两种 encoder 实现分别在编译开/关下用 dummy weights，验证编译计数器只在 `compile_mm_encoder` 开启时增加。回归风险：功能默认关闭（gated behind `compile_mm_encoder`），不改变默认行为；本工作与 #56277 的 Pixtral encoder CUDA-graph 支持互补、不重复也不修改其实现，其 pixtral.py / mistral3.py 改动可干净叠加，encoder CUDA-graph capture 前会做 eager warmup 以便先完成编译；正文建议 #56277 合并后再补一个 `compile_mm_encoder=True` 加 `cudagraph_mm_encoder=True` 的集成测试。

## 关联

正文明确 #56277（为 Pixtral encoder 增加 CUDA-graph 支持，与本 PR 互补）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
