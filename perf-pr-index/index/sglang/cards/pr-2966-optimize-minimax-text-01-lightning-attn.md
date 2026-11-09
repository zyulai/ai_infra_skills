# PR #2966: optimize MiniMax-Text-01 lightning_attn_decode triton

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2025-01-18 |
| 分类 | Attention / Kernel(tier: major) |
| Label | (none) |
| Diff | diffstat 未提供(pack 素材无增删行数字) |
| 链接 | https://github.com/sgl-project/sglang/pull/2966 |

## 一句话结论

去掉 lightning_attn_decode Triton kernel 的显式 F.pad,改在 kernel 内用 mask 计算,相对原 PyTorch decode 实现端到端提速 30%-50%。

## 动机

MiniMax-Text-01 的 lightning attention decode 已有 Triton kernel,但要求 head dim(d)与 value dim(e)是 2 的幂。main 分支实现为了满足这一点,在 host 侧对 q/k/v/kv 做 `F.pad` 到 `next_power_of_2`,再逐个 `contiguous()` 并把 kv 转 fp32,引入一整条 padding/拷贝开销;kernel 本身按 padded 满尺寸无边界地 load/store。decode 是逐 token 路径,这些固定开销在小 batch 下占比显著。

## 优化点

`_decode_kernel` 新增 `d_original`/`e_original` constexpr(及 BLOCK_SIZE 参数):kernel 内构造 `d_mask = d_idx < d_original`、`e_mask = e_idx < e_original`,对 q/k/v 用一维 mask load,对 KV 的 d x e 矩阵用二维 mask load/store,输出按 `e_mask` store,padding 区不参与计算。host 侧删除 `F.pad + contiguous + fp32 转换` 链,改为直接分配 padded 形状的空 tensor 后整块切片拷贝(`q_padded[..., :d] = q` 等),消除逐 tensor 的 pad-再拷贝;kv 仍在进入 kernel 前保持 fp32 语义不变。benchmark 的 `batch_size_range` 从 2^0..2^11(最大 2048)收敛为 1..32,匹配 decode 实际 batch 范围。

## 关键改动

```text
benchmark/kernels/minmax-text-01-lighting_attention/benchmark_lighting_attention_decode.py: kernel mask 化 + host 去 F.pad + benchmark 范围调整
```

kernel 内的 mask 化(替代数据级 padding):

```python
# Create masks for original dimensions
d_mask = d_idx < d_original
e_mask = e_idx < e_original
q = tl.load(Q + qk_offset + d_idx, mask=d_mask, other=0.0)
k = tl.load(K + qk_offset + d_idx, mask=d_mask, other=0.0)
v = tl.load(V + v_offset + e_idx, mask=e_mask, other=0.0)
kv = tl.load(
    KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
    mask=(d_mask[:, None] & e_mask[None, :]),
    other=0.0,
)
```

## 性能声明

PR 正文给出 main 与 PR 两份 `lightning-attention-decode-performance` 基准表(batch_size 1-32,seq_len=1,Original PyTorch vs Triton 实现;硬件与数值单位 PR 未注明)。摘点:

- bs=1:Triton 实现 312.096(main)→ 297.376(PR);Original PyTorch 384.128 → 379.136。
- bs=16:338.496 → 300.928(PyTorch 侧 372.736 → 372.736 基本持平)。
- bs=32:466.400 → 307.712;main 分支 Triton 实现随 batch 增大从 312 涨到 466,PR 后稳定在约 290-308。

正文结论:去除显式 padding、在 kernel 内直接用 mask 计算显著降低开销,相比原 PyTorch MiniMax-Text-01 decode 版本取得 **30%-50% end2end speedup**。

## 验证与回归风险

验证:benchmark 文件内含 `test_lightning_attention_implementations`,断言 PyTorch 实现与 Triton 实现的输出及 kv 结果一致(一致时打印 "Two implementations match");`get_benchmark` 改为 bs 1-32、seq_len 1。pack 未含 CI 状态,PR 正文亦无其他测试章节。风险:改动集中于单个 benchmark/kernel 文件的 MiniMax-Text-01 lightning attention decode 路径;d/e 非常规尺寸时仍会 pad 到 2 的幂(空 tensor 加切片拷贝),mask 恒为真时行为等价旧实现;正确性依赖实现一致性测试,pack 未注明更大规模的模型级验证。

## 关联

无(pack 未注明同区域其他 PR;reviewer 为 @zhyncs)。同分类(Attention / Kernel)其他线性注意力/lightning attention 优化可另行检索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
