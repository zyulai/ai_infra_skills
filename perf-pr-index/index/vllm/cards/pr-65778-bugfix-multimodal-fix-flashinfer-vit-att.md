# PR #65778: [Bugfix][Multimodal] Fix FlashInfer ViT attention for GLM-4.1V, GLM-OCR and GLM-5.3-Flash

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-06 |
| 分类 | Attention / Kernel |
| Label | bug, ready, multi-modality, nvidia, glm |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/65778 |

## 一句话结论

修复 GLM 系列视觉编码器 FlashInfer 后端的崩溃与批量错误,默认后端延迟降 1.5%。

## 动机

使用 `--mm-encoder-attn-backend FLASHINFER` 时,GLM-4.1V/4.5V、GLM-OCR 与 GLM-5.3-Flash 的视觉编码器在 main 上直接失败(eager 与 encoder CUDA graph 均崩,断言 `sequence_lengths is not None`)。且若只修断言,批量推理时第一个序列之后的每个序列都会读错 `v`:输出相对 L2 误差达 166% 或直接 NaN;捕获 1024-token encoder graph 会触发 illegal memory access。FlashInfer ViT 路径在这批 GLM 模型上完全不可用。

## 优化点

根因有二,均出在 FlashInfer 的 `flashinfer_wrapper`(cuDNN batch prefill)对输入布局的假设上。其一,该路径需要 `sequence_lengths`,而 GLM 系列的 vision transformer 从未把它传进各 block(GLM-OCR 与 GLM-5.3-Flash 在 eager 模式下甚至不计算它)。其二,`maybe_recompute_cu_seqlens` 生成的 v 偏移按 `3 * hidden_size` 布局,要求 `v` 是 fused qkv 输出的 view(如 Qwen2.5-VL/Qwen3-VL);GLM attention 却把 `v` 做了 `.contiguous()`,token stride 变成 `hidden_size`,导致后续每个序列按 3 倍偏移读 `v`。修复:把 `sequence_lengths` 沿 GLM vision attention/blocks 透传;`.contiguous()` 只保留给 q/k(RoPE 本就会拷贝),`v` 保持为 qkv 的 view;eager 路径在选中 FlashInfer 时构建其专有 metadata(GLM-5.3-Flash 复用 `prepare_encoder_metadata`,GLM-OCR 因移除了 position embeddings 而内联调用 `MMEncoderAttention` 辅助函数)。其他后端代码路径不变。

## 关键改动

```text
vllm/model_executor/models/glm4_1v.py: sequence_lengths 透传,v 保持 qkv view
vllm/model_executor/models/glm_ocr.py: eager 下按后端构建 FlashInfer metadata
vllm/models/glm5next/common/multimodal.py: GLM-5.3-Flash 同样修复 metadata 构建与 v 布局
tests/models/multimodal/generation/test_vit_cudagraph.py: GLM-4.1V 加入 ViT graph 测试
```

```python
q, k = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k))
# Keep v a view into qkv, as FlashInfer's cu_seqlens expect.
v = rearrange(v, "s b ... -> b s ...")
```

## 性能声明

(测试条件:GB200,bf16,TP1;基线为 main @2a777ce26e)

- 默认后端(FlashAttention)输出与 main bitwise 一致,且因 `v` 不再拷贝,encoder latency geomean 降至基线的 0.985(graph replay)/ 0.984(eager),约 1.5% 降低。测量覆盖 34 个 1–200 张图(64–13,800 token)的 batch,每值取 15 次运行中位数,0/34 个 batch 慢超过 2%。
- 正确性修复前后:批量中单图输出与单独推理的误差从 166% relative L2/NaN 降至 ≤0.21%–0.34%(按模型)。
- 修复后 FlashInfer vs FlashAttention(GLM-4.1V):graph replay geomean 1.003,eager 1.041;graph 模式下 1,300 token 以下小 batch FlashInfer 最多慢 3.2%,eager 下 1,100 token 以下多数 batch 慢 12–16%(per-call host overhead 主导)。本 PR 是正确性修复,不使 FlashInfer 变快。

## 验证与回归风险

新增 GLM-4.1V 到 `test_eonly_vit_cudagraph_outputs`(main 上因断言失败,本 PR 后通过);`pytest tests/v1/cudagraph/test_encoder_cudagraph.py` 78 passed;pre-commit 通过。encoder CUDA graph 与 eager 误差 ≤0.20%,大 batch 弄脏 capture buffer 后重放输出一致。风险:GLM-5.3-Flash 的 encoder CUDA graph 未端到端验证(完整 checkpoint 不可得,用真实 config + 种子随机视觉权重替代);改动限定 GLM 视觉路径,其他后端走原路径,默认后端输出不变。

## 关联

#65453(对齐 FlashInfer `cu_seqlens` 的 capture/replay,与本 PR 触碰不同行、可组合,graph 结果在含/不含它时均成立);#49852(encoder CUDA-graph 路径,`prepare_encoder_metadata` 的来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
