# PR #65987: [Bugfix] Make fused MoE expert weights contiguous before GPU copy in Llama-4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yuvalluria |
| 合并 | 2026-11-10 |
| 分类 | MoE(tier: major) |
| Label | bug, llama |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65987 |

## 一句话结论

Llama-4 fused MoE expert weights 拷 GPU 前补 `.contiguous()`,H100 NVL 合成基准传输提速 19.87x。

## 动机

`load_moe_expert_weights` 中,经 transpose/chunk/indexing 得到的 fused MoE expert weight view 在仍非 contiguous 的状态下被传给 `weight_loader()` 并拷贝到 GPU;非 contiguous CPU tensor 的 DMA 传输远慢于 contiguous tensor,拖慢权重加载。Fixes #31624。

## 优化点

在 `weight_loader()` 调用前、对已切片的最终 piece 加 `.contiguous()`,而不是对整个 weight 提前调用:此前的修复尝试 #32081(因 stale 自动关闭,并非被否决)发现 blanket `.contiguous()` 会 regress 较小的 tensor。Scope note 说明:在 H100 NVL 上用合成基准测试了三处可能有同样问题的位置,只有 fused MoE expert weight 路径有明确的大幅改善;FP8 block-scale 与 Q/K rotary permute tensors 在各自较小尺寸下单独加 `.contiguous()` 反而更慢,因此被刻意排除,留待更精确的真实尺寸数据。由于 MoE expert weights 主导 checkpoint 总字节,这是收益已被测量验证的那一处。

## 关键改动

```text
vllm/model_executor/models/llama4.py: load_moe_expert_weights 的 fused 分支在 weight_loader() 之前对最终切片 tensor 调 .contiguous(),附注释说明原因
```

```python
if fused and isinstance(new_loaded_weight, torch.Tensor):
    new_loaded_weight = new_loaded_weight.contiguous()
```

## 性能声明

合成基准(非端到端),H100 NVL,torch 2.6.0a0+df5bbc09d1.nv24.11;对象为 transpose(-1,-2) 后的 basic-indexed view,shape [64,4096,1024]:

- non-contiguous transfer:mean=574.401 ms,min=536.849 ms,max=585.899 ms
- contiguous() transfer:mean=28.911 ms,min=24.788 ms,max=103.387 ms
- speedup:19.87x(同 shape 首次运行为 22.35x,作者归因 GPU/CPU 调度噪声)

PR 未注明端到端模型加载提速;真实 Llama-4 checkpoints(109B-400B)单卡放不下,故采用合成基准。

## 验证与回归风险

PR 未列新增单测,验证即上述传输微基准。回归风险限于 `llama4.py` 的 fused MoE 权重加载路径(仅 `fused=True` 分支);在最终切片 tensor 上调用,避免拷贝放大;FP8 block-scale 与 rotary permute 路径未被改动。

## 关联

Fixes #31624;前次尝试 #32081(stale 自动关闭)。无其他同区域 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
