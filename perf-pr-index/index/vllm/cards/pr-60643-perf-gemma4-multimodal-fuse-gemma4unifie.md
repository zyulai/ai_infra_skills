# PR #60643: [Perf][Gemma4][Multimodal] Fuse Gemma4Unified 2D posemb and batch image/video embedder

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lucianommartins |
| 合并 | 2026-10-22 |
| 分类 | Model-specific Fast Path(major) |
| Label | ready, multi-modality |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含模型文件与对应测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/60643 |

## 一句话结论

向量化 Gemma4Unified 2D 因式分解位置嵌入并把多图/多帧视频嵌入批量化,A100 上组件级提速 1.23–6.94x,E2E 1.01–1.04x,greedy token 100% 一致。

## 动机

gemma4_unified.py(仅 google/gemma-4-12B-it 使用的 encoder-free 视觉路径)的 _factorized_posemb 用逐轴循环加多次 3D 索引,在 280–2240 视觉 token 上产生约 0.31 ms 的小 kernel 启动开销;_process_image_input/_process_video_input 对每张图/每帧做 Python 循环逐个投影,8 帧视频触发 8 次 vision embedder 与 8 次 ColumnParallelLinear 投影,SM 利用率低。

## 优化点

一是 _factorized_posemb 向量化:pos_embedding view(-1, embed_dim) 后用单次 F.embedding 查表(x_flat=px*2、y_flat=py*2+1,clamp -1),in-place add_ 累加两轴,masked_fill_ 把 padding 位置置零,保持 bit-exact,不依赖 Triton/自定义 kernel。二是批量嵌入路径 _embed_vision_patches:3D 张量或变长 list 先 cat 成 [total_patches, D],一次 vision_embedder 加一次 embed_vision GEMM,再按长度 split 并去除 padding;_vision_chunk 按可用显存预算分块(镜像 _encoder_chunk 逻辑,total_items<=1 短路,显存查询包在 gpu_sync_allowed() 内以兼容异步执行与 CUDA graph);视频按 video_frame_counts split。改动严格限于 Gemma4Unified,tower 型 Gemma4(gemma4_mm.py)不受影响。

## 关键改动

```text
vllm/model_executor/models/gemma4_unified.py: _factorized_posemb 向量化 +
  _embed_vision_patches 批量化/_vision_chunk 显存分块
tests/models/multimodal/processing/test_gemma4_unified.py: 新增 parity、
  chunking、CUDA 单测
```

## 性能声明

A100-SXM4-80GB,main(0838cd6593) vs feature 分支:_factorized_posemb 1.25–1.38x(280 tokens 0.310→0.248 ms;2240 tokens 0.325→0.236 ms),bit-exact;_process_image_input 1/2/4/8/16 图为 1.23x/2.03x/2.71x/3.30x/4.28x(16 图 18.125→4.234 ms),变长 8 图 4.46x;_process_video_input 1 视频 4/8 帧 3.92x/5.47x,2 视频 16 帧 6.32x,4 视频 32 帧 6.82x,8 视频 64 帧 6.94x(73.23→10.56 ms)。E2E(gemma-4-12B-it,V1,torch.compile+CUDA Graphs):prefill bs=8 408.97→393.92 ms,mm_serving_e2e bs=8 1076.41→1031.63 ms,整体 1.01–1.04x,greedy token 100% 匹配。

## 验证与回归风险

pytest tests/models/multimodal/processing/test_gemma4_unified.py 20 passed 19 skipped;ruff check/format 通过;16 个 microbenchmark case 无 parity 错误。数值:posemb 100% bit-exact(max_abs_diff=0);小批量 bit-exact,大批量 cos sim ≥0.99999(归因 BF16 GEMM batching 结合性)。风险面限于 Gemma4Unified 的视觉输入处理;文本路径与 tower 型 Gemma 4(E2B/E4B/26B-A4B/31B)走 gemma4_mm.py,不经过改动代码。

## 关联

无(pack 未列出关联 PR;经 GitHub 搜索确认无重复工作)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
