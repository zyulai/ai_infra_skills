# PR #34580: Flashinfer cuDNN backend for Qwen3 VL ViT attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @maxyanghu |
| 合并 | 2026-02-27 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, ci/build, v1, multi-modality, qwen, nvidia |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 7 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/34580 |

## 一句话结论

为 Qwen3 VL ViT 编码器接入 FlashInfer cuDNN attention 后端,GB200 上 encoder forward 均值快 19.3%。

## 动机

多模态 encoder 的 ViT attention 此前支持 FLASH_ATTN、TRITON_ATTN、TORCH_SDPA 等后端,缺少 FlashInfer cuDNN 路径。NVIDIA MLPerf Qwen3 VL MoE 提交团队在跑分中发现 cuDNN backend 对该 ViT 形状更快,故将其作为 `MMEncoderAttention` 的新后端接入,通过 `--mm-encoder-attn-backend=FLASHINFER` 启用。注意仅支持 Qwen3 VL ViT,Qwen2.5 VL ViT 不支持。

## 优化点

新增 FLASHINFER 后端走 `cudnn_batch_prefill_with_kv_cache`:cuDNN 后端要求传入 batch offsets 与实际序列长度,因此在 vision blocks 调用前新增 `cu_seqlens`、`sequence_lengths`、`max_seqlen` 的计算(Numpy 侧完成再异步上 GPU)。由于 cuDNN frontend graph 以 batch size 和 max sequence length 作为 cache key,为避免运行时反复重编译 graph,对三者做 bucket padding:batch 桶 [8,16,32,64],max_seqlen 桶 1K-128K。据 cuDNN 团队测量,桶间 kernel 性能无显著差异,bucketing 只是为省掉内存操作开销(TODO 注明待 issue #34763 的 CUDA graph 支持后移除)。此外预分配 128 MiB 全局 workspace buffer 复用;q/k 强制 contiguous(cuDNN <= 9.10.2.21 的要求,对带 RoPE 的 ViT 无额外成本);`compute_max_seqlen`/`maybe_compute_sequence_lengths`/`maybe_recompute_cu_seqlens` 逻辑从 qwen3_vl 模型代码上提到 `MMEncoderAttention` 供各模型复用。

## 关键改动

```text
vllm/model_executor/layers/attention/mm_encoder_attention.py: 新增 FLASHINFER 分支、bucket padding 与 cu_seqlens 重算工具方法、128 MiB workspace
vllm/v1/attention/ops/vit_attn_wrappers.py: 新增 flashinfer_wrapper 自定义 op(封装 cudnn_batch_prefill_with_kv_cache)及 fake impl
vllm/model_executor/models/qwen3_vl.py: ViT forward 前计算并传递 sequence_lengths/cu_seqlens/max_seqlen,删除原 compute_attn_mask_seqlen
vllm/platforms/cuda.py: get_supported_vit_attn_backends 加入 FLASHINFER
tests/kernels/attention/test_mha_attn.py: 新增 head_size=72 的 varlen forward FLASHINFER 对照测试
```

## 性能声明

GB200,Qwen/Qwen3-VL-30B-A3B-Instruct,`vllm bench mm-processor --num-prompt 1000`,负载为 1024 text tokens + 1 张图(50% 256x256、50% 720x1280)再生成 128 tokens,1000 次 encoder 调用:encoder_forward_ms 均值 FLASH_ATTN 7.16 vs FLASHINFER 5.78,mm encoder performance gain = 19.3%;端到端延迟均值 25937.46 vs 24701.19 ms(中位数 27018.73 vs 25512.37 ms)。PR 未注明其他精度设置。

## 验证与回归风险

测试计划:`vllm bench mm-processor` 基准;新增 `tests/kernels/attention/test_mha_attn.py:test_mha_attn_varlen_forward_flashinfer`(bf16/half,varlen,与参考实现 assert_close atol/rtol 1e-2,通过 monkey-patch model_config 模拟 FLASHINFER 后端选择)。风险面:改动了 `MMEncoderAttention` 公共 forward 签名(sequence_lengths 参数)与 qwen2_5_vl/qwen3_vl 两个模型的 ViT 调用路径,其他 multimodal 模型若复用该层需兼容;cu_seqlens 重算仅在 FLASHINFER 后端生效,原 FA/TRITON 路径行为保持;依赖 flashinfer 包(importorskip)与 cuDNN 版本(q,k contiguous 要求)。

## 关联

同区域:多模态 encoder attention 后端体系(FLASH_ATTN/TRITON_ATTN/TORCH_SDPA/ROCM_AITER_FA 现有后端)。PR 注明 TODO 依赖 issue #34763(cuDNN backend 的 CUDA graph 支持),落地后可移除 bucket padding。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
