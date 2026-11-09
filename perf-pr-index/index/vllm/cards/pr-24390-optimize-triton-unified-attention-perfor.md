# PR #24390: Optimize triton unified attention performance for sliding window attention

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zixi-qi |
| 合并 | 2025-09-19 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用,单文件小改动) |
| 链接 | https://github.com/vllm-project/vllm/pull/24390 |

## 一句话结论

triton unified attention 的 2D kernel 对 sliding window 注意力做 tile 剪枝:只遍历窗口覆盖的 KV tile,不再对被 mask 掉的序列白算 attention score。

## 动机

GPT-OSS 这类带 alternating sliding-window 层的模型走 triton attention 后端时,`kernel_unified_attention_2d` 对每个 Q-block 仍从头到尾遍历全部 prefix tile(`for j in range(0, num_tiles)`),窗口外的 key 位置靠 mask 丢弃——计算照做、结果全弃。profiling 显示全局注意力层与滑动窗口层耗时几乎相同:局部注意力层的算力被 mask 浪费,窗口越长(相对序列)浪费比例越高。

## 优化点

在 kernel 内做解析式 tile 剪枝:对当前 Q-block,先求出其覆盖的查询位置区间 `qpos_lo = q_block_local_idx * BLOCK_Q`、`qpos_hi = min(qpos_lo + (BLOCK_M-1)//num_queries_per_kv, cur_batch_query_len - 1)`;由滑动窗口定义,每个查询 q 只能看到 `[q_abs - SLIDING_WINDOW + 1, q_abs]`(q_abs = context_len + q),于是整个 Q-block 允许的 key 区间为 `[context_len + qpos_lo - SW + 1, context_len + qpos_hi]`;换算成 tile 下标并 clamp:`tile_start = max(0, first_allowed_key // TILE_SIZE)`,`tile_end = min((last_allowed_key // TILE_SIZE) + 1, num_tiles)`,循环改为 `for j in range(tile_start, tile_end)`。`SLIDING_WINDOW <= 0` 时保持原全局行为,零额外开销;全局注意力层完全不受影响。改动仅约 25 行,无新 kernel、无配置变化。

## 关键改动

```text
vllm/attention/ops/triton_unified_attention.py: kernel_unified_attention_2d 增加 sliding-window tile 剪枝
tests/kernels/attention/test_triton_unified_attention.py: sliding_window 参数扩展为 [None, 64, 128, 256]
```

剪枝核心(节选):

```python
first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
last_allowed_key = context_len + qpos_hi
tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
for j in range(tile_start, tile_end):
```

## 性能声明

PR 正文无量化数字(regex 检测为 none),提供 before/after profiling 截图各一张(GPT-OSS + triton attention):改前全局与局部注意力层耗时相当,改后滑动窗口注意力层"much cheaper",具体百分比未给出。精度验证:GPT-OSS 120b(medium reasoning,temp 1.0)AIME25 eval 得分 0.875(std 0.3307),未附基线对照。单元测试 64 passed、64 skipped。PR 未注明测试条件(硬件型号未写明)。

## 验证与回归风险

单测参数化扩展:sliding_window 从 `[None, 256]` 增至 `[None, 64, 128, 256]`,与 ref_paged_attn 参考实现对照,覆盖更小窗口(即更大剪枝幅度)的边界情况。回归风险面很小:逻辑仅在 `SLIDING_WINDOW > 0` 分支生效,剪枝错误(如 qpos_hi 上界、跨 tile 边界的 key 区间换算)会直接表现为精度回归,被参数化单测覆盖;非滑动窗口模型与其它 attention backend 不受影响。AIME25 单点 eval 提供端到端 sanity 但无统计力。

## 关联

前序: triton unified attention kernel 的引入 PR(编号未在素材中,索引另有 pr-20308 prefill 优化、pr-18218 causal-conv 等同 kernel 家族卡)。同区域:GPT-OSS 滑窗层相关优化。本批次内同分类卡: pr-58769(Kimi-K3 Triton descriptor 迁移)、pr-50572(FlashInfer BF16 GEMM)。
