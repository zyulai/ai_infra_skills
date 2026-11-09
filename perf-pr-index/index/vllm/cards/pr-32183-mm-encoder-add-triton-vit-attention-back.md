# PR #32183: [MM Encoder] Add Triton ViT attention backend

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Isotr0py |
| 合并 | 2026-02-15 |
| 分类 | Attention / Kernel |
| Label | performance, rocm, ready, v1, qwen, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None, None files);diff 含约 13 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/32183 |

## 一句话结论

为多模态 encoder(ViT)attention 新增 Triton backend,替代 FP32 ViT 下低效的 TORCH SDPA,Qwen3-VL 4B 视频输入 e2e benchmark 中 Mean TTFT 降约 3.8%。

## 动机

`MMEncoderAttention` 目前在 ViT 使用 FP32 时依赖 TORCH SDPA,而 SDPA 对这种场景效率不高。作者为 `MMEncoderAttention` 增加一个 Triton attention backend 作为更快的替代;复用已有的 `triton_prefill_attention.context_attention_fwd`(非因果 attention),无需新写 kernel。

## 优化点

在 `vllm/v1/attention/ops/vit_attn_wrappers.py` 新增 `triton_attn_wrapper` custom op(含 fake impl 注册):把 4D QKV 展平为 `(b*s)` 布局,用 `cu_seqlens` 构造 varlen 分段,调用 `context_attention_fwd(is_causal=False)` 后还原形状。`mm_encoder_attention.py` 新增 `_forward_triton` 分支并接入 `forward_cuda` 分发。后端选择逻辑重写:`vllm/platforms/cuda.py` 的 `get_vit_attn_backend` 从"只试 FlashAttention"改为按优先级列表(FLASH_ATTN → TRITON_ATTN → TORCH_SDPA)逐一尝试 `supports_head_size`/`supports_dtype`/`supports_compute_capability`,使 head_size=72 这类不能被 FA 支持的 FP32 ViT 自动落到 Triton;CUDA 与 ROCm 的 supported VIT backend 列表均加入 TRITON_ATTN。同时给 qwen2_vl/qwen2_5_vl/qwen3_vl/qwen3_omni/glm4_1v/ernie45_vl/dots_ocr/paddleocr_vl 等模型的 `compute_attn_mask_seqlen` 加入 TRITON_ATTN 分支,并移除 Qwen2.5-VL/Qwen3-VL/PaddleOCR-VL 构造函数中的 backend 白名单报错。

## 关键改动

```text
vllm/v1/attention/ops/vit_attn_wrappers.py: 新增 triton_attn_wrapper custom op,包装 context_attention_fwd
vllm/model_executor/layers/attention/mm_encoder_attention.py: 新增 _forward_triton 分发分支
vllm/platforms/cuda.py: ViT backend 选择改为按支持性逐个尝试,TRITON_ATTN 列为第二优先
vllm/platforms/rocm.py: supported VIT backends 加入 TRITON_ATTN
tests/kernels/attention/test_mha_attn.py: head_size=72 FP32 断言选 TRITON_ATTN;数值容差 rtol=atol=1e-3
```

wrapper 核心调用:

```python
q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
output = torch.empty_like(q)
context_attention_fwd(
    q, k, v, output,
    b_start_loc=cu_seqlens[:-1],
    b_seq_len=cu_seqlens[1:] - cu_seqlens[:-1],
    max_input_len=max_seqlen,
    is_causal=False,
    softmax_scale=scale,
)
```

## 性能声明

测试条件:PR 正文未注明 GPU/硬件。`vllm bench serve`,Qwen3-VL-4B-Instruct,random-mm 数据集,100 prompts,max-concurrency 64,`--random-prefix-len 25 --random-input-len 300 --random-output-len 40`,每请求 1 个视频(无图片),分辨率桶 (640,360,1200),`--request-rate inf --ignore-eos --seed 42`。Triton Attn vs SDPA:

- Benchmark duration 340.21 vs 351.83 s;输出吞吐 11.70 vs 11.32 tok/s;Peak output token throughput 149.00 vs 123.00 tok/s。
- Mean TTFT 162125.75 vs 168587.35 ms(约 -3.8%),Median TTFT 147391.71 vs 164011.89 ms。
- Mean TPOT 303.19 vs 308.27 ms;Mean ITL 632.51 vs 599.94 ms(ITL 略差)。

## 验证与回归风险

依赖测试 `tests/kernels/attention/test_mha_attn.py`:新增 CUDA + head_size=72 + FP32 断言后端选中 TRITON_ATTN;Triton 路径数值对比放宽到 `rtol=1e-3, atol=1e-3`,其余后端保持默认严格容差——本身即说明精度特性不同。风险面:`get_vit_attn_backend` 选择顺序变化会影响所有多模态模型的 ViT 后端自动选择(FA 可用时不影响);`vision.py` 中异常捕获从 `AssertionError` 扩为 `(AssertionError, AttributeError)`;多个 VL 模型移除了构造期的 backend 白名单硬校验,改为运行时分支。

## 关联

同区域为 ViT/encoder attention 提供后端选择的前序机制:`vit_flash_attn_wrapper` 与 `vit_torch_sdpa_wrapper`(本 PR 在同一文件并列新增第三个 wrapper)。pack 未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
