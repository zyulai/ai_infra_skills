# PR #61458: [Perf][Model] Batch PaddleOCR-VL vision encoder and projector

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ssingh231219 |
| 合并 | 2026-10-14 |
| 分类 | Other（tier: major） |
| Label | ready, multi-modality |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/61458 |

## 一句话结论

PaddleOCR-VL 视觉编码器/projector 从逐图调用改为批量一趟，EngineCore 每 crop CPU 时间 −46%，吞吐 +34–85%。

## 动机

PaddleOCR-VL 原实现对每个调度到的 image 各跑一次 vision encoder 和 projector。serving 负载下 EngineCore 是 CPU-bound（base 与本 PR 都约跑满一个核），逐图的调度/CPU 开销直接限制吞吐。把多张图合并成一次 encoder/projector pass 并消除逐层 device-to-host 同步，就能减少每 crop 的 CPU 工作。

## 优化点

- Encoder：对 processor 输出的 flat patch tensor 一次调用覆盖全部调度图像；position 坐标每图重新起始，累积 `cu_seqlens`（每图一个 segment）保证 attention 不跨图。
- Projector：normalization、linear、GELU 是 per-token 操作，改为在拼接后的全部特征上一次执行；依赖各图 grid 的 2×2 spatial merge 仍逐图做，输出按原图序切分回去。
- 消除 per-layer device sync：FlashAttention 的 `max_seqlen` 改为在 H2D 拷贝前用 CPU 侧 `image_grid_thw` 元数据计算（attention wrapper 会对它调 `.item()`，若为 device tensor 每层都会同步）。
- 输入布局：`pixel_values` 从 per-image batched 字段改为 `flat_from_sizes`，与 flat patch 张量配套。batch 大小由现有多模态 scheduler 与 encoder budget 决定，峰值激活显存仍受 profiled encoder budget 约束。

## 关键改动

```text
vllm/model_executor/models/paddleocr_vl.py: 唯一改动文件——encoder/projector 批量化 + 去同步
tests/models/multimodal/test_paddleocr_vl.py: 新增 7 个单测
```

 projector 中 per-token 与 per-image 计算的拆分：

```python
# Norm and MLP are per-token, so run them once over all images;
# only the 2x2 merge depends on each image's grid.
sizes = [image_feature.shape[0] for image_feature in image_features]
normed_features = self.pre_norm(torch.cat(image_features)).split(sizes)
```

 encoder 侧跨图隔离与去同步：

```python
# One segment per image so that images never attend to each other.
cu_seqlens = torch.from_numpy(np.cumsum([0, *seq_lens], dtype=np.int32))
...
# Computed before the H2D copy: the attention wrappers call `.item()`
# on it, which would sync on every layer for a device tensor.
```

## 性能声明

（H100 PCIe，PaddleOCR-VL-1.6 BF16，base `d9a543c`，两版本同环境/flags/输入。）

- OmniDocBench（19,251 regions / 1,651 pages，C=64）：Throughput 44.70→82.84 crops/s（**+85.3%**）；EngineCore CPU 23.13→12.45 ms/crop（**−46.2%**）；API server CPU 持平；Text/Formula NED、Table TEDS 均在 BF16 噪声内持平；EngineCore 两版本都约 103% 单核，完成 workload 用时 431 s→232 s。
- `vllm bench serve`（random-mm，64/64 token，每请求 1 图，4 种尺寸均分）：C=32 +34.2%（36.90→49.50 req/s，TTFT 494→303 ms）、C=64 +38.9%、C=128 +47.0%、C=256 +42.3%（40.89→58.18 req/s）；12,288 个请求全部成功。
- PR 结论：EngineCore CPU 每 crop 降 46%，吞吐视 workload 提升 34–85%；OmniDocBench 增益更大，因其 crop 多为小文本区域，单步能塞更多图，逐图开销占比更高。

## 验证与回归风险

- 新增 7 个单测：patch packing、attention offsets、positions、per-image spatial merging、与逐图路径的 FP32 parity、BF16 对 FP32 精度、跨图隔离；既有 4 个 MRoPE 测试通过。
- `VLLM_GPU_SYNC_CHECK=error` 下 TP=1/TP=2 serving 32 个 public crops 零错误、零被标记同步；TP=2 输出对比 32/32 与基线完全一致。
- 风险：只覆盖 eager 路径（encoder 未包 `@support_torch_compile`、`compile_mm_encoder` 默认 False 时生效）；端到端数字在 `-cc.mode=0`、decode 用 `FULL_DECODE_ONLY` CUDA graph 的配置下测得，默认编译配置下 encoder 代码路径相同但未按该配置测端到端。

## 关联

- PR #44394：encoder CUDA graph capture/replay + packed projector 路径；其 eager fallback 仍逐图调用 encoder，本 PR 修的正是该 eager 路径（同文件、不同代码路径）。
- 同区域：本批 PR #22742（Qwen2.5VL ViT 数据并行，同为多模态 vision 路径优化）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
