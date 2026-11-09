# PR #36527: MiniMax-M3: share the sparse index top-k across layers and reuse the decode top-k buffer

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zcnrex |
| 合并 | 2026-09-08 |
| 分类 | Attention / Kernel(tier: major) |
| Label | amd, run-ci, jit-kernel |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36527 |

## 一句话结论

MiniMax-M3 稀疏注意力跨层共享 index top-k 并复用 decode top-k 缓冲,8x MI350X 上 input 吞吐 +10.04%–11.49%。

## 动机

MiniMax-M3 的稀疏注意力在每个 sparse layer 都要跑一次 lightning-indexer pass 选出 top-k KV blocks。在 value-disabled sparse layers 上 indexer 没有自己的输出(`idx_o` 为 None,只消费 block selection),其结果本可被连续若干层共享,但此前每层重复计算;decode 时每层还为 top-k 结果新分配张量,CUDA-graph 下也存在分配问题。

## 优化点

新增 `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=N`(默认 2),将每 N 个 disable-value sparse layers 分组:组内第一层计算并存储 reduced top-k,其余层复用,跳过 flash-index attention 与 group-reduce。decode 的选择结果直接写入按 batch size 持久化的 `int32` 缓冲(`topk_out=`)而非每层新分配,目标地址固定使 CUDA-graph capture 不再分配。由于共享 selection 会改变 skip 层关注的 KV blocks(正确性敏感),该特性被严格门控:非 ROCm 平台及启用 two-batch overlap 时 `index_topk_freq` 固定为 1;freq=1 时不分配缓冲、新参数全为 None/False,CUDA 路径与 main 逐字节一致。顺带改动:prefill 的 layer 无关 seqblock 元数据(`cu_seqblocks_q` 三元组)改为每 ForwardBatch 计算一次下传(缓存 key 含 owning ForwardBatch 对象,兼容 two-batch overlap 拆分);`forward_decode` 补上 extend 路径已有的 `_is_sparse_kv_cached_by_fusion` 守卫,避免 fused qk-norm-rope 已写入时重复 sparse-KV store;decode score kernel 的 autotune 剪掉 `BLOCK_SIZE_N < block_size` 的配置,ROCm 上改用固定配置 + BLOCK_SIZE_N 启发式(runtime autotune 可能在 graph capture 下 mistune)。

## 关键改动

```text
python/sglang/srt/layers/attention/minimax_sparse_backend.py: 分组共享 top-k、持久 decode 缓冲、seqblock 元数据缓存
python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py: prefill/decode 支持 cached_topk_idx / topk_out
python/sglang/srt/environ.py: 新增 SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ(默认 2)
python/sglang/kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py: autotune 剪枝与 ROCm 固定配置
```

```python
self.index_topk_freq = (
    max(int(envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.get()), 1)
    if is_hip() and is_gfx95_supported() and not is_tbo_enabled()
    else 1
)
self.index_cache_enabled = self.index_topk_freq > 1
```

## 性能声明

测试条件:8x MI350X(gfx950),MiniMax-M3-MXFP8,TP8,80k input / 600 output,cache hit rate 90%,baseline 为 upstream/main,每 boot 两轮取 warm rep(两轮差 <0.1%),噪声 ±0.4%。input throughput +10.04%–11.49%(bs=1:187984→209588 tok/s;bs=32:193245→215010);output throughput +6.38%–20.59%(bs=1:55.35→59.67;bs=32:934.92→1127.4)。accuracy:gsm8k 512 examples(score)97.07%→96.68%。注:PR 说明 `--moe-runner-backend aiter` 在当前 main 对 mxfp8 不生效、实际回落 triton,A/B 两侧同为 triton MoE runner,对比公平;aiter MoE 路径并非被测对象。

## 验证与回归风险

gsm8k 512 examples(`max_tokens 2048`,`temperature 0`,`seed 0`)与 perf 同一 server boot。共享 selection 会改变 skip 层关注的 KV blocks,故仅在 ROCm(gfx95)且非 two-batch overlap 时生效;freq=1 时与非改动路径逐字节一致,其他平台零影响;freq 越大复用越陈旧的 selection,2 为 accuracy-safe 默认。decode 缓冲在 graph capture 之外按 bs 惰性预分配。pack 中 CI 状态:Extra 与 AMD ROCm 7.2 run 均显示失败标记。

## 关联

正文未引用其他 PR 编号;特性与 two-batch overlap 互斥(代码门控)。同批索引中 #33480(AMD prefill CP two-batch overlap)、#36505(ROCm aiter gfx950 prefill)属同一 AMD/ROCm 优化区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
