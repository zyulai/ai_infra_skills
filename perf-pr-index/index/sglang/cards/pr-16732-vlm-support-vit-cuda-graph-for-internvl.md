# PR #16732: [VLM] Support ViT CUDA Graph for InternVL

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-01-14 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance, Multi-modal, run-ci, vlm |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 internvl.py、qwen2_5_vl.py 修改与新增 internvl_vit_cuda_graph_runner.py |
| 链接 | https://github.com/sgl-project/sglang/pull/16732 |

## 一句话结论

为 InternVL 家族新增专用 ViT CUDA Graph Runner,E2E 性能提升约 5%,精度无下降。

## 动机

此前 Qwen2.5-VL / Qwen3-VL 已有 ViT CUDA Graph 支持,但它们设置 `flatten_batch=True`,而 InternVL 模型的 `flatten_batch` 为 False,现有 ViTCudaGraphRunner 无法直接复用,导致 InternVL 的 vision encoder(多模态 prefill 的关键耗时段)始终以 eager 方式逐层执行,实现复杂度因此增加,需要专用的 graph runner。

## 优化点

新增 `InternViTCudaGraphRunner`:不做 batch 展平,输入保持 `[B,S,H]` 三维,graph key 为 `(B, S)` 二元组(而非 token 数);cu_seqlen 按连续布局构造为 `[0, S, 2S, ..., B*S]`。runner 为每个 key 维护地址稳定的 input/output buffer、attention workspace(`[tokens, nheads, head_dim]`)与 cu_seqlens buffer,先做一次小规模 eager warmup 触发惰性初始化,再整体捕获"全部 encoder layer 的前向"为一张 CUDA graph;运行时只 copy 输入并 replay。整图捕获意味着 attention 与 MLP 全部进入 graph,消除逐层 kernel launch 开销。启用开关为环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH`,启用时不再创建 aux_stream。仅支持 `triton_attn` 与 `fa3` 两种 mm attention backend,其他 backend 显式报错。顺带把 Qwen2.5-VL 的 graph runner 构造也改为按同一开关懒创建。

## 关键改动

```text
python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py: 新增,InternVL 专用 ViT CUDA graph runner,(B,S) 为 key、整图捕获全部 encoder layers
python/sglang/srt/models/internvl.py: encoder 构造 runner,forward 在 enable_cg 且非 output_hidden_states 时走 graph 路径;attention 层透传 output_ws workspace
python/sglang/srt/models/qwen2_5_vl.py: ViTCudaGraphRunner 改为 enable_cg 时才创建,统一开关逻辑
```

```python
def _graph_key(self, x: torch.Tensor) -> Tuple[int, int]:
    # x: [B,S,H]
    return (x.shape[0], x.shape[1])

def _build_cu(self, B: int, S: int, device: torch.device) -> torch.Tensor:
    # [0, S, 2S, ..., B*S]
    return torch.arange(0, (B + 1) * S, step=S, device=device, dtype=torch.int32)
```

## 性能声明

- **E2E 性能提升约 5%**(作者正文;PR 附两张 profiler 截图,未在文本中给出具体分段数字,也未注明模型/硬件/workload 条件)。
- 精度:作者声明 "No acc drop",并用 lmms-eval 在 `OpenGVLab/InternVL3_5-8B` 上跑 `mmmu_val`(900 样本,batch_size 16)给出 mmmu_acc **0.5689**。

## 验证与回归风险

验证依赖 lmms-eval 的 mmmu_val 评估结果;未附端到端吞吐 benchmark 表格。回归风险:graph 路径只返回 last_hidden_state,`output_hidden_states=True` 时回落 eager;仅 triton_attn/fa3 backend 可用,其他 mm_attention_backend 配置会 RuntimeError;`max_len` 在 warmup/capture 时通过 `.item()` 读回 host,graph key 为精确 (B,S) 组合,输入尺寸组合多时首个请求会有捕获开销;Qwen2.5-VL 路径被同步改动,行为由环境变量开关控制。

## 关联

同区域:Qwen2.5-VL/Qwen3-VL 的 ViT CUDA Graph(本 PR 明确以其为前序实现,并复用 `ViTCudaGraphRunner` 思路);CUDA Graph / Compiler 分类下的其他 graph 捕获 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
