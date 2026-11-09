# PR #22424: [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @1am9trash |
| 合并 | 2026-04-09 |
| 分类 | Attention / Kernel |
| Label | (无) |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/22424 |

## 一句话结论

HIP LayerNorm 改用 aiter CK layernorm2d,NSA indexer 的 k_norm 每 call 3 kernel 合为 1,~12μs→~4μs。

## 动机

HIP 上 `LayerNorm` 走 torch 实现,入口与出口各触发一次 dtype cast,每次调用共 3 个 kernel(cast→layernorm→cast),拖慢 GLM-5-FP8 NSA indexer 里的 `k_norm()` 等操作的性能。

## 优化点

`LayerNorm.forward_hip()` 在输入 dtype 为 bf16/fp16 且与层 dtype 一致时改调 aiter 的 CK kernel `layernorm2d_fwd()`:先把 x reshape 成 2D 计算、再 view 回原 shape;其他 dtype 维持原 torch 路径。配套把 NSA indexer 的 `k_norm` dtype 在 `SGLANG_USE_AITER`+HIP 时从 fp32 改为 bf16,使其能命中 CK 路径(此前 k_norm 用 fp32 LayerNorm,无法受益)。机制上是把 cast+norm+cast 三 kernel 合并为单 kernel 调用。

## 关键改动

```text
python/sglang/srt/layers/layernorm.py: forward_hip 增加 aiter layernorm2d_fwd 分支(bf16/fp16 且 dtype 匹配)
python/sglang/srt/layers/attention/nsa/nsa_indexer.py: k_norm dtype fp32→bf16(aiter 启用时)
```

```python
if (_has_aiter_layer_norm
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == self.dtype):
    orig_shape = x.shape
    x = x.reshape(-1, self.hidden_size)
    return layer_norm(x, self.weight, self.bias, self.variance_epsilon).view(orig_shape)
```

## 性能声明

MI355X TP8、GLM-5-FP8(nsa tilelang prefill/decode backend、fp8_e4m3 KV、disable-radix-cache),concurrency 4/8/16/32/64 平均,baseline 为 sglang PR #22258 + aiter PR #2575:
- ISL/OSL 1k/1k:throughput +1.4%,TPOT −0.9%
- ISL/OSL 8k/1k:throughput +1.2%,TPOT −2.8%

Per-layer profiling:单次 LayerNorm 时间约 12μs→4μs,kernel 数 3→1。

## 验证与回归风险

`python/sglang/test/test_layernorm.py::TestLayerNorm` 全部 384 个 subtest 通过;模型级验证 GLM-5-FP8 MI355 GSM8k(TP8)0.946。风险:`k_norm` dtype 从 fp32 降为 bf16 属数值精度变化(由 GSM8k 覆盖);非 bf16/fp16 或 dtype 不匹配的输入回落 native;收益限于 HIP+aiter 环境,CUDA/NPU 不受影响;所有走 `forward_hip` 的 LayerNorm 都会切到 CK kernel,属广覆盖改动,非仅 NSA 路径。

## 关联

baseline 依赖 sglang PR #22258 与 aiter PR #2575(正文注明);同区域其他 PR 编号正文未列。无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
