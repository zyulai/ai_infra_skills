# PR #46955: [Perf] dsv41: Use the Engram all-to-all lookup for decode and CUDA graphs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-10-28 |
| 分类 | MoE |
| Label | documentation, deepseek |
| Diff | diffstat 缺失(素材未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/46955 |

## 一句话结论

Engram DP lookup 的 all_to_all 扩展到 decode 与 CUDA graph,decode 自 DP 4 起最多快 48%。

## 动机

这是 #46917 与 #46925 的 follow-up。此前 Engram 的 lookup=`auto` 只对 eager prefill 选择 all_to_all,decode 批次与 CUDA graph replay 一律沿用 reduce-scatter;decode microbenchmark 显示 all_to_all 在多数 shape 更快。另外 auto 的选择此前每步在 `_dp_exchange` 里现算,应收敛到并行布局确定后一次解析。

## 优化点

新增 `resolve_engram_config`:进程内一次、模型构造时把 auto 解析为具体 lookup(条件 is_cuda、tp>1、attn_dp_size==tp、attn_cp_size==1、hash 列可整除 TP 组→ALL_TO_ALL,否则 ALL_REDUCE),结果记录进 engram_config 并打日志 `Engram lookup auto -> ...`,每步的 `_dp_exchange` 被删除。DP lookup 拆成 `_dp_reduce_scatter_lookup` 与 `_dp_all_to_all_lookup` 两个方法;等 per-rank counts(MAX_LEN batches、CUDA graph)改用 fixed-size collective(`reduce_scatter_tensor` / `group.all_to_all_single`)以便被 graph capture,不等 counts 保留 variable-size 版本;`_dp_sizes` 相应调整(MAX_LEN 且全局 buffer 长度匹配时返回等值 counts)。行为变化:DP-attention decode(CUDA,列可整除)从 reduce-scatter 换成 all-to-all,输出 bitwise 相同。

## 关键改动

```text
python/sglang/srt/layers/engram.py: resolve_engram_config 一次解析 auto;_dp_all_to_all_lookup/_dp_reduce_scatter_lookup;_dp_sizes 支持 graph capture 下的等值 counts
docs/docs/advanced_features/engram.mdx: auto 的新行为与 decode 测量文档
```

```python
def resolve_engram_config(columns: int) -> EngramConfig:
    config = get_exec().offload.engram_config or EngramConfig()
    if config.lookup != EngramLookup.AUTO:
        return config
    parallel = get_parallel()
    tp = parallel.tp_size
    lookup = EngramLookup.ALL_REDUCE
    if (is_cuda() and tp > 1 and parallel.attn_dp_size == tp
            and parallel.attn_cp_size == 1
            and cols_per_rank(columns, tp) is not None):
        lookup = EngramLookup.ALL_TO_ALL
```

## 性能声明

GB300,一层 host-table Engram,rank 1,us/layer:DP4 decode 400 tokens:reduce_scatter 101.2 vs all_to_all 69.2;DP4 decode 1,280:187.1 vs 130.5;DP4 prefill 32K:reduce_scatter 3,819.1 vs all_to_all 2,463.2;DP2 decode 16:28.2 vs 37.3。decode microbenchmark(GB300/B300/H200,DP 2-8,16-1,280 tokens/rank,CUDA graphs):all_to_all 在 48 个 shape 中 35 个更快;从 DP 4 起最多比 decode 原用的 reduce-scatter 快 48%,≤64 tokens/rank 时最多落后 12us;DP 2 时最多落后 20us、最多领先 53us。PR 未注明端到端 serving 数字。

## 验证与回归风险

GB300 DP2/DP4 上对 EngramEmbedding DP lookup 做 bitwise 检查(prefill uniform/skewed/单 rank 为空;decode 在 CUDA graph 中 capture 后用新 ids replay),各选择均与 all_reduce 相等;test_deepseek_v41_flash_gb300_dp2ep2.py 通过(GSM8K 0.98,MMMU-Pro 0.79);attention DP4/EP4 recipe GSM8K 0.985。风险面:行为变化限定 CUDA + DP-attention decode + 列可整除;不等 counts 的 graph capture prefill 仍走旧 collective;HIP、attn_cp_size>1、world_dp_gather_enabled 时退回 global-buffer 路径。

## 关联

#46917、#46925(前序)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
