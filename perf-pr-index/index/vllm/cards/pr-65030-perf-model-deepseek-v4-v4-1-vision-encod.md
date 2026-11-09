# PR #65030: [Perf][Model] DeepSeek V4/V4.1 Vision: encode a call's images in packed passes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @subhashpolisetti |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, multi-modality, deepseek, DSv4, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/65030 |

## 一句话结论

DeepSeek V4/V4.1 Vision 多图改一次 packed 编码，H100 上 encoder 提速 2.0–2.6× / 1.6–2.1×。

## 动机

Overview 指出 DeepSeek V4/V4.1 Vision 对每张图各跑一次 ViT + aligner，即使一次调用携带多张图；`_process_image_input` 逐图循环，每张图都要为 32 个 block 各自的内核启动付费。V4.1 的 encoder CUDA graphs 已使用 `forward_packed` 路径。本 PR 把一个 call 里的所有图在一次 packed pass 中编码，无需新增 flag。PR 正文称 per-image 路径受 kernel launch 限制、对主机敏感。

## 优化点

在 `common/vision.py` 新增 `encode_images()`：当 `len(vit_grid) > 1` 且 attention backend 为 `FLASH_ATTN` 时，用 `build_packed_vit_metadata` / `build_packed_merge_metadata` 构造 `cu_seqlens` 等元数据，调用 `vision_model.forward_packed`（以 `cu_seqlens` 做 per-image attention）一次跑完所有图，再经 `aligner.forward_packed` 并按每图行数 `split` 回列表；单图或非 FlashAttention 回退逐图路径。`common/vl_model.py` 与 `deepseek_v41/nvidia`、`deepseek_v41/amd` 的 `vl_model.py` 删除逐图循环/`_encode_image`，改调 `encode_images`。`run_dp_sharded_vision_tower`、graph manager 的 eager fallback 不变。代价：每个 matmul 见到更多行导致舍入不同、输出非 bit-identical，peak encoder 显存随图数增长。

## 关键改动

```text
vllm/models/deepseek_v4/common/vision.py: 新增 encode_images()，FlashAttention+多图走 forward_packed。
vllm/models/deepseek_v4/common/vl_model.py: _process_image_input 以 encode_images 替换逐图循环。
deepseek_v41/nvidia/vl_model.py、deepseek_v41/amd/vl_model.py: 同样改调 encode_images。

  attn_backend = vision_model.blocks[0].attn.attn.attn_backend
  if len(vit_grid) == 1 or attn_backend != AttentionBackendEnum.FLASH_ATTN:
      return [aligner(vision_model(item, h, w), h, w) for item, (h, w) in ...]
  hidden_states = vision_model.forward_packed(
      patches, vit_cos, vit_sin, cu_seqlens, max_seqlen)
  rows = aligner.forward_packed(hidden_states, merge_idx, merge_mask)
```

## 性能声明

Encoder（H100、FlashAttention、真实 tower 权重、VisionArena 尺寸随机 patches，54 个多图调用、3 轮、打乱顺序）per-image → this PR：2 图 V4 59.6→29.9 ms、V4.1 59.6→30.1；4 图 119.2→47.0 / 118.8→57.9；8 图 237.6→88.8 / 236.3→101.8；16 图 483.0→172.2 / 481.6→219.2；32 图 962.9→328.4 / 961.2→418.6 ms；全部 54 调用合计 2.61× / 2.10×。Claims：encoding 在两台 H100 上对 V4 快 2.0–2.6×、V4.1 快 1.6–2.1×，无调用慢于 2%；cap 8 图的首版为 2.55×/2.08×，另一台 H100 为 2.01×/1.63×（2 图 per-image 44.9 ms）。Serving（真实 V4 checkpoint、2× B200、TP2+EP、FP8 KV cache、block 256、greedy、thinking off、无 spec decode、max_tokens=32）：4 图 c1 中位 TTFT 378/367→271 ms、req/s 2.51/2.56→3.44；VisionArena c16 中位 TTFT 959/914→548 ms、req/s 10.0/16.0→24.6（0 failed requests）。MMMU-Pro（1730 题、138 多图、concurrency 32）正确数 861/860→862、多图 74/75→75，与 main 的分歧 200 vs 96 of 1730。Peak 显存 32 图调用最多增约 1.7 GB(V4)/2.0 GB(V4.1)。所有运行基于 `190df0b900` main wheel + 本 PR 文件。

## 验证与回归风险

单元测试按要求移除（曾在 H100 比对 packed 与 per-image 输出并通过），精度由上述 MMMU-Pro 覆盖；encoder-level 对 FP32 per-image 参考的误差与 cap 版一致（V4 packed vs per-image 差 0.39/1.33/0.16%，V4.1 0.04/0.16/0.24%）。Startup KV-cache 基本不变（202,321/202,340 → 202,335 tokens）。风险与边界：输出非 bit-identical、PR 与 main 分歧多于 main 自比（200 vs 96）且双向；peak encoder 显存随图数增长；packing 仅作用于 FlashAttention 且 ≥2 图，单图与其他 backend 仍走 per-image。Not tested：V4.1 serving/MMMU-Pro（仅 encoder 结果）、AMD（共用 helper）、thinking mode、DSpark speculative decoding、更大输出预算的 MMMU-Pro。

## 关联

PR 正文 Overlaps 明确提及：#62802（已合并，本分支仍可干净合并）、#57985（触及这些文件、可干净合并，移除 patch dtype cast）、#58782（在 `amd/vl_model.py` 冲突）、#64435 与 #57872（同样针对 V4.1 但不触及这些文件）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
