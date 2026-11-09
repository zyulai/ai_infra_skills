# PR #24701: [Kernel] [CPU] refactor `cpu_attn.py:_run_sdpa_forward` for better memory access

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ignaciosica |
| 合并 | 2025-09-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 1 个文件(实际移动 4 行代码) |
| 链接 | https://github.com/vllm-project/vllm/pull/24701 |

## 一句话结论

CPU SDPA 路径把 GQA 的 repeat_interleave 挪到 movedim 之后,沿 head-major 轴复制得到连续内存,注意力计算快 2 倍以上。

## 动机

CPU 注意力后端 `_run_sdpa_forward` 对 GQA 模型(num_kv_heads != num_heads)在做 `movedim` 重排之前就对 K/V 执行 `repeat_interleave` 复制。此时张量仍处于原始布局,复制后得到的是大跨度 strided view,后续 `scaled_dot_product_attention`(CPU flash kernel)消费前被迫做昂贵的整块拷贝,内存访问模式很不友好。

## 优化点

机制是调整两个张量变形算子的顺序:先 `movedim` 把 K/V 对齐到 `[B, H, L, D]` 布局,再 `repeat_interleave`(沿 head-major 的 `dim=-3` 轴)物化多出的 query heads。这样复制发生在已经 head-major 的连续维度上,产出天然连续的内存布局,消除 SDPA 路径上的隐式大拷贝。数学完全不变(同样的复制语义),纯顺序调整,只影响 GQA 配置;MHA(num_kv_heads == num_heads)路径完全不执行该分支。

## 关键改动

```text
vllm/v1/attention/backends/cpu_attn.py: _run_sdpa_forward 中 repeat_interleave 移到 movedim 之后
```

```python
key = key.movedim(0, key.dim() - 2)
value = value.movedim(0, value.dim() - 2)
if self.num_kv_heads != self.num_heads:
    key = key.repeat_interleave(self.num_queries_per_kv, dim=-3)
    value = value.repeat_interleave(self.num_queries_per_kv, dim=-3)
```

## 性能声明

本地 benchmark(多种负载场景,具体硬件未注明):注意力计算 **快 2 倍以上,输出完全一致**。离线生成 profile 对照:Qwen3-0.6B、1188 token prompt、生成 512 token、temperature 0——`aten::scaled_dot_product_attention` 17.884 s → 5.530 s;整个生成过程 Self CPU time 45.337 s → 32.625 s;`repeat_interleave`/`clone` 调用耗时约减半。两次运行输出文本逐字一致。

## 验证与回归风险

无新增单测:PR 正文明确说作者初次参与、希望社区指导如何正确单测该改动;验证手段是 master 与 PR 分支的 profiler 对照加输出比对。风险点:仅覆盖 CPU(SDPA)后端;依赖 PyTorch CPU flash kernel 对连续输入的快路径,若上游行为变化收益可能缩水;GQA 复制维度从 `dim=1` 改为 `dim=-3` 语义等价(均为 head 轴),但解码/编码器等不同 `AttentionType` 分支共用此函数,需确认各分支布局假设一致。

## 关联

同区域(CPU attention 优化):#25948(CPU unquantized linear 加速)与 #29193(CPU paged attention GEMM 优化)为同一 CPU 后端的其他热点;pack 未提供直接依赖 PR。
