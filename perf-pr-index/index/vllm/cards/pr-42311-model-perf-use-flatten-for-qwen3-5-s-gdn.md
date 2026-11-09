# PR #42311: [Model] [Perf] Use flatten for Qwen3.5's GDN output projection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @rishaps |
| 合并 | 2026-05-18 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | ready, qwen |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 仅含 vllm/model_executor/layers/mamba/gdn_linear_attn.py 三处单行替换 |
| 链接 | https://github.com/vllm-project/vllm/pull/42311 |

## 一句话结论

Qwen3.5 GDN 输出投影把 einops rearrange 换成 tensor.flatten(-2),消除每次 GDN forward 的 Python 字符串解析开销,单次调用平均快 8.23x(CUDA)。

## 动机

Qwen3.5 的 GDN(gated delta net)输出投影用 einops 的 `rearrange(core_attn_out, "... h d -> ... (h d)")` 合并最后两维。einops 每次调用都要做 Python 层的 pattern 字符串解析与校验,在每次 GDN forward 都重复发生,属于纯 host 侧开销;PyTorch 原生 `flatten(-2)` 语义等价且几乎无解析成本。

## 优化点

单行语义等价替换,共三处:CUDA 路径 _output_projection、forward_xpu、forward_cpu 中的 `rearrange(..., "... h d -> ... (h d)")` 全部改为 `core_attn_out.flatten(-2)`(注释保留映射关系)。机制上没有算法变化,收益全部来自绕开 einops 的 Python 字符串解析/校验路径,直接调 aten::flatten。

## 关键改动

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py: _output_projection(CUDA)/forward_xpu/forward_cpu 三处
  rearrange(core_attn_out, "... h d -> ... (h d)") -> core_attn_out.flatten(-2)
```

## 性能声明

Profiler repro 模型 Qwen/Qwen3.5-0.8B(vllm bench latency + torch profiler,enforce-eager,batch 1)。CUDA eager(dtype fp16,max-model-len 64,input 16/output 32):einops rearrange 576 次调用平均 23.261us,aten::flatten 平均 2.828us,per-call 平均 speedup 8.23x。CPU eager(dtype fp32,input 8/output 8):41.741us vs 1.949us,per-call speedup 21.41x。正确性(PPL 对比 HF):einops 19.509801864624023 vs flatten 19.510005950927734,相对 HF expected 19.38858413696289 的偏差分别为 0.6252% 与 0.6263%。E2E 吞吐数字未提供(收益为 host 侧调用开销)。

## 验证与回归风险

Correctness test:tests/models/language/generation_ppl_test/test_qwen.py(Qwen3.5-0.8B 需 language_model_only、cpu_offload_gb=10、kv_cache_memory_bytes 256M 的 tweak)。回归面:仅 Qwen3.5 GDN 层的三个设备路径(CUDA/XPU/CPU)输出投影前的 shape 合并,数值语义等价(view/reshape 类操作),风险极低;若其他模型代码复用同一 einops pattern 则不在本 PR 范围内。

## 关联

同区域:Qwen3.5 模型支持 PR #34110(本索引库同批次,同属 Qwen3.5/GDN 路径)。
