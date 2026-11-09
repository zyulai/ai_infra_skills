# PR #11722: Support mrope triton kernel and add unit test

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-10-20 |
| 分类 | Attention / Kernel(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供 +N/-N);pack diff 片段含 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/11722 |

## 一句话结论

为 Qwen-VL 系列的 mrope 引入 Triton kernel,rotary embedding 计算提速 32%-48%。

## 动机

`MRotaryEmbedding` 的多模态(2D positions)路径此前依赖 torch.compile 的 native 实现,开销偏高。本 PR 引入专门的 Triton mrope kernel(改编自 vLLM,代码注释指向 Liger Kernel 的 qwen2vl_mrope),对 Qwen2-VL / Qwen2.5-VL 的 q/k 做 in-place 旋转,降低 rotary embedding 在多模态 prefill/decode 中的耗时,并补齐 benchmark 与单测。

## 优化点

在 `rotary_embedding.py` 新增 `_triton_mrope_forward`:每个 token 一个 program,按 `(3, num_tokens, head_dim//2)` 的 cos/sin 缓存以 t/h/w 三段 mask 分别加载并合成 `cos_row`/`sin_row`,对 q/k 的左右两半做 `y=[x1,x2]*[cos,cos]+[-x2,x1]*[sin,sin]` 的 in-place 旋转,支持 interleaved mrope。`triton_mrope` 包装函数做 next_power_of_2 pad 与 contiguous 保证。`MRotaryEmbedding.forward` 在 positions 为 2D(多模态)时分派到 `triton_mrope`,1D positions 保持 `_apply_rotary_emb`;原实现改名 `forward_native`,并新增 `_match_cos_sin_cache_dtype` 避免昂贵的 `nn.Module.__setattr__`。同时新增 benchmark 脚本与一致性单测并注册进 run_suite。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding.py: _triton_mrope_forward/triton_mrope 与 forward 分派
sgl-kernel/benchmark/bench_mrope.py: 新增 mrope benchmark 脚本(适配自 vLLM)
test/srt/rotary_embedding/test_mrope.py: 新增 forward_native vs forward 一致性单测
test/srt/run_suite.py: 注册 rotary_embedding/test_mrope.py
```

## 性能声明

Qwen2.5-VL-72B-Instruct、bfloat16、num_tokens 1024、benchmark 100 iters(warmup 10):TP8 speedup 1.32250401x(Torch mean 0.00016325s vs Triton 0.00012344s,即 32%),TP4 1.46622326x(45%),TP2 1.48466421x(48%)。gsm8k(200 题)acc 0.950 no drop;单测 8 passed。GPU 型号未注明。

## 验证与回归风险

`test_mrope.py` 对 Qwen2-VL-7B/72B、Qwen2.5-VL-72B 在 tp_size 1/2、num_tokens 11/8192 下对比 `forward_native` 与 `forward`(atol 1e-2, rtol 1.6e-2),标记为 CUDA-only。风险:1D positions 分支未走 Triton(保持 native);2D 分支要求 `mrope_section` 存在,否则 assert;cos_sin_cache 的 dtype/device 惰性转换影响所有 mrope 模型。

## 关联

同文件后续 PR #17420(GLM4v `get_rope_index` 热路径优化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
