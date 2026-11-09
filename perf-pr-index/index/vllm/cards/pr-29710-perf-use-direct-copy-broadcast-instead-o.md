# PR #29710: [perf] Use direct copy (broadcast) instead of cat for k_nope/k_pe in MLA prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @minosfuture |
| 合并 | 2025-12-11 |
| 分类 | Attention / Kernel(major) |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 common.py 与新增 benchmark 两个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/29710 |

## 一句话结论

MLA prefill 中 k_nope/k_pe 拼接改为预分配输出 + 两次直接广播拷贝,取代 torch.cat(expand),batch≥512 时平均提速 1.99x(3.16ms→1.61ms @B=32768)。

## 动机

v1 MLA prefill 路径用 `torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)` 拼接 K。对 expand 产生的非连续张量,torch.cat 的处理有性能惩罚;MLA prefill 通常是大 batch,这一步的开销被放大:batch 32768 时 concat 达 3.16 ms(`k=[32768,128,192]`),成为 prefill 的明显热点。

## 优化点

新增 `_concat_k_nope_k_pe` 辅助函数:先 `torch.empty` 预分配形状 `[..., nope_dim + pe_dim]` 的输出,再用两次带广播的直接切片赋值分别写入 k_nope 与 k_pe 部分,跳过 expand + cat 的中间张量与拼接逻辑。共替换三处调用:`_compute_prefill_context`、`_context_parallel_compute_prefill_context`(CP 路径)与 `_forward_prefill`,均为 prefill 上下文构建路径,decode 不涉及。机制上等价、结果相同,只是把"拼接"换成"预分配 + 定向拷贝"。

## 关键改动

```text
vllm/v1/attention/backends/mla/common.py: 新增 _concat_k_nope_k_pe,替换三处 prefill 路径的 torch.cat(expand) 写法
benchmarks/kernels/benchmark_mla_k_concat.py: 新增跨 batch size 的 cat vs direct copy 微基准
```

关键片段(common.py):

```python
k = torch.empty(
    (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
    dtype=k_nope.dtype, device=k_nope.device,
)
k[..., : k_nope.shape[-1]] = k_nope
k[..., k_nope.shape[-1] :] = k_pe
```

## 性能声明

微基准(`benchmark_mla_k_concat.py`,k_nope=[B,128,128]、k_pe=[B,1,64],float8_e4m3fn,bfloat16 同测):
- B=32768:cat 3.159 ms → direct 1.593 ms(1.98x),与 PR 目的陈述的 3.16ms→1.61ms 一致;B=65536:1.99x。
- B≥512 平均 1.99x(B=2048 达 2.14x);B≤256 时直接拷贝反而更慢(0.37x–0.98x,固定 kernel 启动开销占主导)。
- 全范围 summary:Min 0.37x / Max 2.14x / Mean 1.52x;结论是 direct copy 在 B≥512 后占优,而 MLA prefill 典型为大 batch。
- E2E 精度:local-completions gsm8k(nvidia/DeepSeek-R1-0528-FP4,num_concurrent=32,limit 100),flexible-extract 与 strict-match 均 0.96。PR 未注明该 E2E 的硬件与微基准所用 GPU。

## 验证与回归风险

验证方式为 trace 级性能确认、精度验证(gsm8k)与新增微基准;diff 未新增单元测试。风险面:仅影响 v1 MLA common 的 prefill/CP prefill 路径(所有走该 common 实现的 MLA 后端);小 batch prefill 下理论上有轻微回退(微基准显示 B≤256 变慢),decode 与 CUDA-graph 路径不受影响。

## 关联

同区域(MLA prefill/DeepSeek 注意力):#26132(DS MLA 计算优化)、#20034(FlashInfer MLA ragged prefill);本批次同分类(Attention/Kernel)的 #64580 属 DSA indexer 方向。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
