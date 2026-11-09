# PR #43335: [Fix] [AMD] aiter: Fix garbage decode output for non-sliding-window models on the vectorized_5d KV layout

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#43335](https://github.com/sgl-project/sglang/pull/43335) |
| 作者 | jiaqiang-dot-liu |
| 合并日期 | 2026-10-20 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
AITER vectorized_5d KV 布局下非滑窗模型 decode 静默读错 KV 页输出乱码;新增 decode_uses_page_table 标志按布局建页表修复。

## 问题现象
AITER 后端开启 SHUFFLE 5D KV 布局(`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`)后,所有**不带 sliding-window attention** 的模型 decode 静默产出乱码:decode 读到错误的 KV 页,且没有任何报错。MI325X 上 Qwen3-14B-FP8 的 GSM8K 从 0.885 掉到 0.000。自 #27063 引入该布局起即如此。现网 workaround 是设 `SGLANG_USE_AITER_UNIFIED_ATTN=1`。gpt-oss(5D 路径当初为之添加的模型)因强制开 unified attention 而不受影响。

## 根因
正文披露:`forward_decode_vectorized_5d` 把 `forward_metadata.kv_indices` 作为 `block_tables` 传给 `pa_decode_gluon`(2-D `(bs, max_num_blocks_per_seq)` 页表,docstring 也写明 "Full-attn → `kv_indices` page table")。但 decode metadata 只在 unified-attention 分支才构建这张 2-D 页表:

```python
if not self.use_triton_unified_attention:
    # flat per-token kv_indices, addressed through kv_indptr
    ...
else:
    # 2-D page table
    ...
```

`use_triton_unified_attention` 对滑窗模型强制开启,而其余模型默认关闭,于是 `pa_decode_gluon` 把 per-token id 当页 id、按行 stride 为 1 去读。eager 路径、CUDA-graph 页表 buffer 以及 capture/replay 都做了同样的错误选择。

## 修复方案
改 `python/sglang/srt/layers/attention/aiter_backend.py`:decode metadata 格式与 decode kernel 是两个独立选择,故显式命名前者——

```python
self.decode_uses_page_table = (
    self.use_triton_unified_attention or self.kv_cache_is_vectorized_5d
)
```

并在五处"flat indices vs 页表"的判断点改用该标志:eager decode(普通与 speculative)、CUDA-graph 页表 buffer 分配、`_apply_cuda_graph_metadata` 里共享于 capture/replay 的两处判断。`use_triton_unified_attention` 本身不变(强制打开会连 `_use_unified_verify` 的 verify kernel 一起切走,超出本修复范围)。NHD 布局下 `decode_uses_page_table` 等于 `use_triton_unified_attention`,不受影响。

新增 `test/registered/amd/test_aiter_vectorized_5d_decode.py`:默认模型(dense、无滑窗)以 `SGLANG_USE_AITER=1` + 5D 布局 + `--page-size 64` 起 server 跑 GSM8K(阈值 0.7),注册进 `stage-b-test-1-gpu-small-amd-mi35x`。

验证(MI325X, ROCm 7.2, Qwen3-14B-FP8, GSM8K 200 题):5D 布局 0.000 → 本 PR 0.890(与 unified-atten workaround 持平;NHD page-size 64 为 0.885);吞吐与 mean TPOT 无回退。

## 影响与回归风险
- 影响窗口:#27063 引入 vectorized_5d 布局之后、本 PR 之前的版本。
- 回归面:仅 `aiter_backend.py` 的 decode metadata 五个判断点。NHD 布局行为不变(标志取值与原判断一致);5D 布局下 decode metadata 改为始终建 2-D 页表,与 `pa_decode_gluon` 的契约对齐;speculative decode 与 CUDA-graph capture/replay 共用同一标志,语义随之统一。

## 关联
- #27063(引入 5D 布局,bug 起点)
- #29009(`pa_decode_gluon` 支持 Qwen3.5-MoE,依赖同一 `kv_indices` 页表契约)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
