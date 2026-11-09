# PR #55285: [Perf] Use SDPA for BLIP-2 Q-Former attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-09-06 |
| 分类 | Attention / Kernel |
| Label | ready, multi-modality |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,仅 1 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/55285 |

## 一句话结论

BLIP-2 Q-Former 自注意力/交叉注意力改用 SDPA 融合内核,Q-Former 几何均值提速 1.43–1.48x。

## 动机

BLIP-2 的 Q-Former attention 手写实现会 materialize 分数矩阵并分步做缩放、softmax、dropout 与第二次 matmul(每层 `{matmul: 2, mul: 1, softmax: 1, dropout: 1}`)。这既多出 score 尺寸的中间张量(显存峰值高),又多出多次内核 launch;PyTorch 的 `scaled_dot_product_attention` 已能融合这一切并自动分派后端内核,Q-Former 路径却没有用上。

## 优化点

把 `vllm/model_executor/models/blip2.py` 中 Q-Former attention 的手写 matmul→scale→softmax→dropout→matmul 链替换为一次 `F.scaled_dot_product_attention`,self-attention 与 cross-attention 都走该路径;显式传 `scale=self.scaling`、训练时传 `dropout_p` 保持原语义。由此消除 score 尺寸的中间张量以及 scale、softmax、dropout、value-matmul 的独立 launch,变成单调用 `{scaled_dot_product_attention: 1}`。纯 PyTorch 可移植路径:无自定义内核、无架构 gate、无 device 分支、无设备专属调优。

## 关键改动

```text
vllm/model_executor/models/blip2.py: Q-Former attention 换成 F.scaled_dot_product_attention
```

```python
context_layer = F.scaled_dot_product_attention(
    query_layer,
    key_layer,
    value_layer,
    dropout_p=self.dropout.p if self.training else 0.0,
    scale=self.scaling,
)
```

## 性能声明

(RTX 4090,CUDA 13.0,PyTorch 2.13.0+cu130;构造真实 `Blip2ForConditionalGeneration` + BLIP-2 兼容维度,经真实 `embed_multimodal` 边界测;确定性有限权重;25 warmup、200 迭代、5 次交替重复取 p50)

| 范围 | Dtype/Batch | Old p50 (us) | New p50 (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| Q-Former | FP16 b1 / b4 | 1519.28 / 1521.30 | 1049.77 / 1006.97 | 1.45x / 1.51x |
| Q-Former | BF16 b1 / b4 | 1504.88 / 1565.29 | 1052.73 / 1088.26 | 1.43x / 1.44x |
| embed_multimodal | FP16 b1 / b4 | 2797.99 / 2744.05 | 2322.34 / 2196.03 | 1.20x / 1.25x |
| embed_multimodal | BF16 b1 / b4 | 2691.73 / 2748.79 | 2248.67 / 2177.32 | 1.20x / 1.26x |

- Q-Former 几何均值:FP16 1.48x,BF16 1.43x。
- Q-Former 峰值额外分配:batch 1 从 1,579,008 降至 1,237,504 字节;batch 4 约 7.3 MiB 降至 5.1 MiB。
- Profiler:每 10 次 Q-Former 调用由 80 次 `aten::bmm` + 40 次独立 softmax 变为 40 次 `aten::scaled_dot_product_attention`,全部分派到 `aten::_flash_attention_forward`。

## 验证与回归风险

作者跑通数值、有限值、可重复性、shape 与输入突变五类校验,八行均通过且无性能回退;ruff check/format 与 `git diff --check` 通过。局限(PR 自述):结果来自单张 RTX 4090、确定性权重,非完整 pretrained checkpoint,也非端到端请求吞吐;不对其他架构/设备/PyTorch 版本做性能声明。BLIP-2 未实现 `SupportsEncoderCudaGraph`,运行在默认 eager 多模态 encoder 路径,不做 CUDA graph 声明;若日后支持 encoder graph,启用 capture 前应重新测 batch-1 短 key 形状。回归面极小:改动仅 BLIP-2 一个文件,语义由 scale/dropout 参数显式保持。

## 关联

无(pack 素材中未列出关联 PR;作者检索了开放的 BLIP-2/Q-Former/SDPA 相关 PR,未发现重叠)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
