# PR #55629: [Perf] Fuse DeepEncoder relative bias in Triton attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-09-08 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, multi-modality |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/55629 |

## 一句话结论

用 Triton attention kernel 在线融合 DeepEncoder 相对位置偏置,替代先展开 dense [B,heads,HW,HW] bias 再 SDPA,64x64 global attention 省下 384 MiB 临时显存、算子延迟约减半。

## 动机

DeepEncoder 的相对位置注意力此前把分解的 rel_h、rel_w 相加并 view 成 dense 4D attn bias 喂给 SDPA;64x64 global attention 因此要 materialize 384 MiB 的 BF16 临时张量,多占显存且多一趟读写。

## 优化点

在 `vllm/model_executor/models/deepencoder.py` 新增 `_deepencoder_rel_pos_attention_kernel`:flash-attention 式 online softmax 前向(Q 按 BLOCK_M 分块、K/V tile 流式、exp2 + running max/sum),在算 QK^T 分数时直接按 `key_h = idx // width`、`key_w = idx % width` 索引并加上 `rel_h + rel_w`,不再构造 dense bias。主机侧 `deepencoder_rel_pos_attention` 以 BLOCK_M=64、BLOCK_N=64、num_warps=4、num_stages=2 启动。分发条件:`use_triton_attention = window_size == 0` 且 CUDA-alike 平台;window attention、CPU、不支持的 dtype/head_dim 保留 SDPA 路径;明确不使用 FlexAttention 或 torch.compile。

## 关键改动

```text
vllm/model_executor/models/deepencoder.py: 新增 Triton fused kernel 与 deepencoder_rel_pos_attention 包装;RelPosAttention.forward 按平台/窗口尺寸分发
```

关键片段(kernel 内在线加偏置):

```python
key_h = key_indices // key_width
key_w = key_indices % key_width
...
rel_h = tl.load(rel_h_ptr + rel_h_offsets, mask=score_mask, other=0.0)
rel_w = tl.load(rel_w_ptr + rel_w_offsets, mask=score_mask, other=0.0)
scores += (rel_h + rel_w) * LOG2E
scores = tl.where(score_mask, scores, -float("inf"))
```

## 性能声明

RTX 4090,CUDA 13.2,PyTorch 2.13.0+cu132,Triton 3.7.1,BF16,50 次交错采样取中位数:attention 算子 p50 1.1453→0.5478ms(降 52.2%),增量峰值显存 390→6 MiB;完整 RelPosAttention 层 p50 1.5667→0.9820ms(降 37.3%),426→42 MiB;max abs error 9.77e-4(算子)/4.88e-4(整层)。此前 FlexAttention 候选整层为 1.4469ms,Triton 版低约 32.1% 且无 torch.compile 进程依赖。E2E 正确性:DeepSeek-OCR 1024x1024 图、greedy、enforce_eager 下生成 21 个 token ID 与基线完全一致。作者注明这是单卡 RTX 4090 的隔离算子结果,非 serving 吞吐声明,不外推到其他 GPU 架构。

## 验证与回归风险

`tests/models/multimodal/test_deepencoder.py` 16 passed、`test_deepseek_ocr.py` 6 passed;覆盖 FP16/BF16 数值等价(方/非方网格)、partial tiles、CPU fallback、混设备输入拒绝、global/window 分发与 14x14/64x64 CUDA Graph capture/replay。回归面:仅 DeepEncoder global attention 的 CUDA FP16/BF16 head_dim=64 路径,其余配置走 SDPA;前向 only。

## 关联

正文提及曾有 FlexAttention 候选实现(整层 1.4469ms,被本实现取代);素材未列出其他同区域 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
