# PR #19003: [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-24 |
| 分类 | MoE(tier: major,classify.py 给出) |
| Label | performance, Multi-modal, run-ci, vlm, flashinfer |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/19003 |

## 一句话结论

为 Qwen3-VL 的 ViT 引入 FlashInfer CUDNN prefill 注意力后端,TTFT 降低 11.6%。

## 动机

FlashInfer 的 CUDNN Prefill 在 ViT attention 上表现出色,本 PR 把它接入 SGLang 作为 VLM ViT 注意力后端:新增 `mm_attention_backend` 选项 `flashinfer_cudnn`,本 PR 先支持 Qwen3-VL,正文声明后续 PR 再适配更多 VLM。

## 优化点

`vision.py` 新增 `VisionFlashInferAttention`:调用 `flashinfer.prefill.cudnn_batch_prefill_with_kv_cache`(causal=False、`is_cuda_graph_compatible=True`),以 packed element indptr(qk/v/o 三段、共 3*(B+1))与 cuDNN 风格 `(B,1,1,1)` 的 actual_seq_lens 传参,并使用 128MB 的常驻 workspace buffer。为减少 cuDNN 计算图重建,batch size 按 `[8,16,32,64]` bucket、max seqlen 按 `[4K..128K]` bucket 化(`qwen3_vl.py` 中对 indptr 与 seq_lens 做补零 pad);`Qwen3_VisionModel` 统一分配 workspace、构建 packed offsets 与 padded sequence_lengths,并新增 `fast_pos_embed_interpolate_from_list`。`server_args.py` 将 `flashinfer_cudnn` 加入 `--mm-attention-backend` 选项。

## 关键改动

```text
python/sglang/srt/layers/attention/vision.py: VisionFlashInferAttention 与 bucket/workspace 常量
python/sglang/srt/models/qwen3_vl.py: packed indptr/seq_lens 构建、workspace buffer、pos-embed 插值
python/sglang/srt/server_args.py: --mm-attention-backend 增加 flashinfer_cudnn
test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py: MMMU 回归测试(阈值 0.51)
```

## 性能声明

Qwen/Qwen3-VL-30B-A3B-Instruct、TP=4、`--mm-attention-backend flashinfer_cudnn`、`--disable-radix-cache`;bench_serving image 数据集(256 prompts、input 128 / output 32、560x560 jpeg、max-concurrency 32):FlashInfer CUDNN mean TTFT 931.84ms vs FA3 1054.52ms(正文:"The performance improved 11.6% vs FA3. (TTFT reduce)");mean TPOT 77.37 vs 71.16ms 略高;mean E2E latency 2360.65 vs 2382.67ms。

## 验证与回归风险

新增 nightly 测试走 lmms-eval 的 mmmu_val,Qwen3-VL-30B-A3B-Instruct 阈值 0.51。风险:`flashinfer_cudnn` 仅 CUDA 路径可用且要求 flashinfer 提供 cudnn prefill API;128MB workspace buffer 常驻显存;qwen3_vl 的 cu_seqlens 构建重构为 numpy 计算再转 torch,NPU 保持 cpu indptr 分支,其他 mm backend 的行为应保持但路径被触碰。

## 关联

同分类(pack 分类为 MoE)另有 PR #28662;正文声明后续 PR 将把该后端适配到更多 VLM。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
