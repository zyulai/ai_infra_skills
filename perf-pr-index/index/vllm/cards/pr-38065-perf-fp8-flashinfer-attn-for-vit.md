# PR #38065: [Perf] FP8 FlashInfer Attn for ViT

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhandaz |
| 合并 | 2026-04-27 |
| 分类 | Quantization(tier: major) |
| Label | documentation, performance, ready, v1, multi-modality, qwen, nvidia |
| Diff | pack 未提供 diffstat(additions/deletions/files 均为 None);diff 另有 12 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/38065 |

## 一句话结论

通过 FlashInfer cuDNN backend 为 ViT encoder attention 增加可选 FP8 量化(Q/K/V 在线量化,动态或静态 scale):GB300 核心注意力 kernel 提速 1.42x,4K 分辨率端到端 encoder forward 提速 1.18x,面向大图 + NVFP4 文本模型场景。

## 动机

对大图、短文本生成的 visual understanding workload,ViT encoder attention 成为显著瓶颈,尤其当文本模型已被量化(如 nvfp4)时。ViT 一般不做 PTQ,因此 Q/K/V 需要在进入 cuDNN attention 前在线量化。本 PR 是 NVIDIA MLPerf Inference Submission V6.0 工作的一部分(动态 scaling 由 @ybgao-nvidia 贡献)。

## 优化点

新增配置 `--mm-encoder-attn-dtype fp8 --mm-encoder-attn-backend FLASHINFER`(要求 FlashInfer cuDNN backend + cuDNN >= 9.17.1)。缩放有两条路:默认动态 scaling——16 项 circular amax history 记录观测到的 Q/K/V amax 并 per-forward 更新 scale,免校准但有额外开销;或静态 per-layer scales——从校准 JSON 文件加载(`--mm-encoder-fp8-scale-path`),并支持动态校准后自动保存(`--mm-encoder-fp8-scale-save-path`,margin 默认 1.5 倍安全余量),官方推荐"先校准、静态运行"。head_dim 非 16 倍数(如 Qwen3-VL 的 72)时,由 stride-aware Triton kernel `quantize_fp8_pad_head_dim_triton` 在量化同时把 head_dim pad 到 16 的倍数,attention 输出再 un-pad(支持非连续 QKV view 与 4D 输入);对齐 head_dim 走 QuantFP8。FP8 模式下 cu_seqlens 需按 `fp8_padded_hidden_size` 重算。明确限制:动态 scaling 暂不兼容 ViT full CUDA graph;目前仅支持 Qwen3-VL 家族(qwen3_vl、qwen3_vl_moe、qwen3_5、qwen3_5_moe);不应默认开启(3 个量化 kernel launch + un-pad 的开销)。

## 关键改动

```text
vllm/config/multimodal.py, vllm/config/model.py: mm_encoder_attn_dtype / fp8 scale 路径与 margin 配置
vllm/model_executor/layers/attention/mm_encoder_attention.py: scale 加载、FP8 量化 dispatch、动态 scaling、自动保存、FP8-aware cu_seqlens
vllm/kernels/triton/qkv_padded_fp8_quant.py: stride-aware FP8 量化 + head_dim pad 到 16 倍数
vllm/v1/attention/ops/vit_attn_wrappers.py: q/k/v_scale、o_data_type 传入 cudnn_batch_prefill_with_kv_cache
vllm/utils/flashinfer.py: is_flashinfer_cudnn_fp8_prefill_attn_supported() 平台/cuDNN 版本检查
docs/features/quantization/fp8_vit_attn.md: 新文档
```

## 性能声明

核心 cuDNN kernel(PyTorch profiler,`cudnn_generated_fort_native_sdpa_sm100_flash_fprop`,head_dim=128、seq_len=8192、QKV [1,8192,16,128]):GB200 350→312 µs(1.12x);GB300 300→211 µs(1.42x)。完整 attention 模块(量化 + cuDNN attention + un-pad,`bench_gpu_time_with_cudagraph`,GB200,head_dim=72):seq 4096:106.5→116.7 µs(0.91x,变慢);8192:377.0→364.6 µs(1.03x);16384:1380.8→1152.3 µs(1.20x)。端到端 encoder forward(Qwen3-VL-30B-A3B-Instruct,GB200,3 images/request):HD 31.77→36.39 ms(0.87x);FullHD 57.99→58.73 ms(0.99x);QHD 131.83→122.30 ms(1.08x);4K 543.44→460.31 ms(1.18x)——crossover 约在 FullHD,QHD 及以上 FP8 胜出。精度(ChartQA,Qwen3-VL-8B-Instruct,500 样本):relaxed_accuracy BF16 0.780 / FP8 dynamic 0.776 / FP8 static 0.780;anywhere_accuracy 0.806 / 0.816 / 0.814;exact_match 0.584 / 0.582 / 0.578(VisionArena 校准的静态 scale 跨数据集保持 BF16 精度)。

## 验证与回归风险

新增 `tests/kernels/core/test_vit_fp8_quant.py`(量化 kernel:连续/非连续 QKV view、4D 输入、pad 区域为零、skip_scale)、`test_vit_fp8_scaling.py`(动态 scale 更新、circular buffer 回绕、静态 JSON 加载、缺层报错、自动保存一次性触发)、`test_vit_fp8_attn.py`(输出形状/un-pad、FP8 vs BF16 closeness:cosine > 0.99、max abs diff < 0.3)。回归影响:opt-in,不影响默认路径;动态 scaling 与 ViT full CUDA graph 不兼容;仅 Qwen3-VL 家族;head_dim 非 16 倍数模型引入 pad/un-pad 与额外 cu_seqlens 重算。

## 关联

NVIDIA MLPerf Inference V6.0 提交的一部分;同分类 Quantization 区域其他相关 PR 未在素材中列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
