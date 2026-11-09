# PR #18185: [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-03-14 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 仅含 python/sglang/srt/models/qwen3_omni_moe.py |
| 链接 | https://github.com/sgl-project/sglang/pull/18185 |

## 一句话结论

优化 Qwen3-Omni Thinker 的 AudioEncoder:FFN 支持并行、向量化 mask 与 conv 快路径降低 CPU 小 tensor 分配压力,ASR 端到端吞吐从 0.28 提升到 3.12 req/s(约 10 倍)。

## 动机

Qwen3-Omni 的 audio encoder 前向在 TP>1 时 FFN 仍走单卡 `nn.Linear`,audio 分支成为 omni 请求的瓶颈;同时 mask 构造按每个长度逐个 `torch.ones` 生成大量小 tensor,conv 阶段无条件分 chunk 循环 + `torch.cat`,CPU allocator 压力与 kernel 数量都偏高,拖慢整体 audio 延迟。

## 优化点

三处机制改动:(1) audio encoder layer 的 `fc1`/`fc2` 从 `nn.Linear` 换成 `ColumnParallelLinear`/`RowParallelLinear`,使 FFN 在 TP>1 下并行分片,前向相应解包返回值;(2) CNN 输出 mask 改为向量化构造——用 `torch.arange(max_len)` 与 `feature_lens_after_cnn` 广播比较一次性生成 bool mask,替代逐长度的 `pad_sequence([torch.ones(...) ...])`;conv 在输入 batch ≤ `conv_chunksize` 时走不分 chunk 的快路径(整块三次 conv+GELU),超过时保留原分 chunk 防止卷积 OOM;(3) `cu_chunk_lens` 构造改为 `aftercnn_lens.tolist()` 一次批量转 Python 并用 `extend` 批量填充,减少逐元素同步;另外 `get_audio_feature` 里 `feature_attention_mask` 显式 `.to(device)` 与输入特征对齐设备。这些都在降低 per-request 的 kernel launch 次数与 CPU 侧小对象分配。

## 关键改动

```text
python/sglang/srt/models/qwen3_omni_moe.py: audio encoder FFN 换并行 Linear;mask 向量化;conv 快路径;cu_chunk_lens 批量构造
```

```python
idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
padded_mask_after_cnn = idx.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)
# conv fast path
if padded_feature.size(0) <= self.conv_chunksize:
    padded_embed = F.gelu(self.conv2d1(padded_feature)); ...
```

## 性能声明

ASR benchmark(Qwen/Qwen3-Omni-30B-A3B-Instruct,`--tp 2`,concurrency 8,100 samples,D4nt3/esb-datasets-earnings22-validation-tiny-filtered,bench_sglang.py):

| 指标 | Main(基线) | 本 PR |
| --- | --- | --- |
| Throughput | 0.28 req/s | **3.12 req/s** |
| Token Throughput | 239.87 tok/s | 659.61 tok/s |
| Average Latency | 5.8475 s | 2.1806 s |
| Median Latency | 1.7992 s | 1.6974 s |
| 95th Latency | 6.1070 s | 5.1574 s |
| Total Test Time | 360.03 s | 32.09 s |

作者结论:audio 性能 **e2e speedup 10x**("PR got 10x improvement")。WER 一项 PR 侧为 858.90、Main 为 3419.81(正文未解释该指标口径;WER 数值异常偏高,仅按原文记录)。

## 验证与回归风险

正确性验证为"audio sample predictions are as expected"(正文附 5 个样本的 REF/PRED 文本对照),无量化精度指标(如 WER 参照集的严格评估)。回归风险:改动限定 Qwen3-Omni 模型文件;conv 快路径以 `conv_chunksize` 为界,batch 超过时行为与原先一致;FFN 并行化要求权重以并行 Linear 加载,TP=1 时行为等价;`feature_attention_mask` 的设备搬运在 CPU mask 输入场景下生效。PR 未附单测/CI 专项测试。

## 关联

同区域:Qwen3-Omni 系列的多模态前向(ViT CUDA Graph 等属其他模型 PR);pack 未列出具体关联 PR 编号,关联暂记"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
