# PR #21126: [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-09-19 |
| 分类 | Attention / Kernel |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/21126 |

## 一句话结论

RotaryEmbedding CUDA 路径可选走 FlashInfer RoPE,microbench 约 2x,端到端无明显收益。

## 动机

vLLM 在 CUDA 上默认用自研 `ops.rotary_embedding` custom op 执行 RoPE。FlashInfer 提供了 `apply_rope_with_cos_sin_cache_inplace` 原地 RoPE 内核,microbenchmark 显示比 vLLM 自定义 op 快约 2x。作者希望将其接入 `RotaryEmbedding.forward_cuda`,为将来默认启用铺路;但端到端收益需先验证。

## 优化点

在 `common.py` 用 `direct_register_custom_op` 注册 `flashinfer_rotary_embedding` custom op:声明 `mutates_args=["query", "key"]`(原地修改 q/k)并配 fake impl,使其兼容 CUDA graph/编译路径;内部调用 FlashInfer 的 `apply_rope_with_cos_sin_cache_inplace`。`base.py` 构造时计算 `use_flashinfer` 开关:custom op 启用、dtype 为 fp16/bf16、CUDA 平台、已安装 FlashInfer、且 head_size ∈ {64, 128, 256, 512}(FlashInfer 内核支持的集合);满足时 `forward_cuda` 直接派发到 FlashInfer op,否则回退原 `ops.rotary_embedding`。走 FlashInfer 时不把 cos_sin_cache cast 成查询 dtype(保留原始 cache);另提取 `_match_cos_sin_cache_dtype` 辅助函数,统一原先散落在 base/DeepSeekScalingRotaryEmbedding/Llama4VisionRotaryEmbedding/MRotaryEmbedding 里的 cache device/dtype 同步逻辑,减少昂贵的 `nn.Module.__setattr__` 调用。默认不启用:PR 正文说明端到端无可见差异,若要默认开启需在 `VllmConfig.__post_init__` 追加 `+rotary_embedding`(正文给出了候选代码但未合入)。

## 关键改动

```text
vllm/model_executor/layers/rotary_embedding/common.py: 注册 flashinfer_rotary_embedding custom op
vllm/model_executor/layers/rotary_embedding/base.py: use_flashinfer 条件与 forward_cuda 派发
vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py: 改用 _match_cos_sin_cache_dtype
vllm/model_executor/layers/rotary_embedding/mrope.py: 同上(forward_native/forward_cuda)
vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py: 同上
```

```python
if self.use_flashinfer:
    torch.ops.vllm.flashinfer_rotary_embedding(positions, query, key,
                                               self.head_size,
                                               self.cos_sin_cache,
                                               self.is_neox_style)
    return query, key
```

## 性能声明

- Microbenchmark:FlashInfer RoPE 相比 vLLM 自定义 op 约 2x speedup(PR 未注明 microbench 的硬件、张量形状等条件)。
- 端到端 lm-eval gsm8k(Llama-3.1-8B-Instruct,5-shot):Torch Native 35.07 it/s(输出 3404.12 toks/s)、vLLM Custom Op 34.66 it/s(3371.77 toks/s)、FlashInfer 33.54 it/s(3261.28 toks/s)——作者结论是"no noticeable performance difference",因此默认不启用。
- 精度:flexible-extract exact_match 分别为 0.7854 / 0.7786 / 0.7817,同量级。

## 验证与回归风险

作者用 lm-eval(llama 8b)做正确性/性能对照,三种实现 gsm8k 结果接近。回归风险主要来自 `use_flashinfer` 条件:head_size 不在 {64,128,256,512}、dtype 非 fp16/bf16 或无 FlashInfer 时自动回退原路径;默认未启用,需显式 custom_ops 开启才生效。`_match_cos_sin_cache_dtype` 重构触及多个 RoPE 变体(DeepSeek scaling、mrope、llama4 vision)的 native 路径,属行为等价替换但覆盖面广。

## 关联

同区域 RoPE 优化:#16457(优化 rotary emb 实现,索引库同分类)。PR 正文未列出其他前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
