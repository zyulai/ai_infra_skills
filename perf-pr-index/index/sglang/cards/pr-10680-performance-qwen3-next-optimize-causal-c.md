# PR #10680: [Performance] Qwen3-Next: optimize causal_conv1d_fn triton kernel - up to 9% faster

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @byjiang1996 |
| 合并 | 2025-09-20 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 3 个文件:hybrid_linear_attn_backend.py、causal_conv1d.py、causal_conv1d_triton.py |
| 链接 | https://github.com/sgl-project/sglang/pull/10680 |

## 一句话结论

Qwen3-Next 的 causal_conv1d_fn Triton kernel 改为 (seq, chunk) 二维 grid,免去 CPU 侧 m_list/offset_list 预计算与 CPU-GPU 同步,kernel 运行时 3X faster,GSM8K 吞吐最高 +8.9%。

## 动机

`causal_conv1d_fn` Triton kernel 用于 extend 与 draft_extend。profile 显示它低效:存在大量 CPU<>GPU copy sync,且 `grid` 方法非常慢——计算 triton grid 需 139us,而真正的 triton kernel launch 本身只有 17us。根因是旧实现要在 CPU 上预先构造每个 BLOCK_M 程序的 (m_list, offset_list) 平铺映射,涉及 numpy/tensor 操作与 H2D 拷贝。

## 优化点

对齐 `triton_backend.py` 的 extend_attention() 的处理逻辑,把 kernel 的程序组织从"平铺 BLOCK_M"改为二维 grid:旧实现中每个 program_id(0) 处理 total BLOCK_M 中的一个 BLOCK_M(需要 batch_ptr/token_chunk_offset_ptr 查表才能映射回 (seq, chunk));新实现中 program_id(0) 直接对应 batch 内一个 seq_len,program_id(1) 对应该序列的一个 token chunk,program_id(2) 沿 feature 维分块。这样 kernel 内部直接 `idx_seq = tl.program_id(0)`、`chunk_offset = tl.program_id(1)`,host 侧完全不需要计算 m_list/offset_list,直接用 `forward_batch.extend_seq_len_cpu` 计算 grid `(len(seq_lens_cpu), ceil(max_seq_len / BLOCK_M), cdiv(dim, BLOCK_N))`,消除了 CPU<>GPU 拷贝与昂贵的 tensor/numpy 预计算;metadata(batch_ptr/token_chunk_offset_ptr/nums_dict)机制整体删除,并附带删除冗余代码/张量、改进 kernel 内 if 条件(增加 `segment_len <= 0` 提前返回)。

## 关键改动

```text
python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py: kernel 签名去掉
  batch_ptr/token_chunk_offset_ptr/batch;grid 改为 (batch, chunks, dim-blocks) 三维;
  host 侧删除 num_program 的 m_list/offset_list 预计算与 metadata 分支
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: forward_extend 调用
  传入 seq_lens_cpu=forward_batch.extend_seq_lens_cpu
python/sglang/srt/layers/attention/mamba/causal_conv1d.py: 接口透传 **kwargs
```

关键摘录(新 grid):

```python
def grid(META):
    max_seq_len = max(seq_lens_cpu)
    return (
        len(seq_lens_cpu),  # batch_size
        (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
        triton.cdiv(dim, META["BLOCK_N"]),
    )
```

## 性能声明

Kernel 运行时:Before p50=313us、avg=656us;After p50=95us、avg=108us(约 **3X faster**;PR 未注明测试硬件与 workload 条件,截图来自 profile)。GSM8K 输出吞吐(标题的 "up to 9%" 对应 TP4 MTP 行):TP4 1613.119→1701.235 tok/s(+5.5%);TP4 MTP 1888.384→2055.889 tok/s(+8.9%);TP4 DP2 MTP 1762.349→1811.621 tok/s(+2.8%)。GSM8K 精度:TP4 0.955→0.950,TP4 MTP 0.955→0.950,TP4 DP2 MTP 0.945→0.940。与 CUDA kernel 对比:改后 Triton 实现与 CUDA 实现在 GSM8K 下相当(2088.367 vs 2091.005 tok/s,精度均 0.945)。

## 验证与回归风险

GSM8K 多配置(含 MTP、DP2)精度与吞吐对照;checklist 勾选了单元测试与文档,但 diff 中未见新增测试文件。回归风险:kernel 签名与调用约定变化(删除 metadata 参数、必填 seq_lens_cpu),所有调用方需同步更新——diff 中更新了 hybrid_linear_attn_backend.py 一处;精度有 0.005 的小幅下降(0.955→0.950 等),PR 未讨论原因。影响路径:Qwen3-Next 等 hybrid linear attention 模型的 extend/draft_extend。

## 关联

修复前序 PR #10552 的 merge conflict(同作者同 kernel);同作者后续 #11536 优化同模型路径的 gpu->cpu sync。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
