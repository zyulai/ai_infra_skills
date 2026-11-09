# PR #8144: [Feature] improve TBO: two chunk overlap

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @House-West |
| 合并 | 2025-08-06 |
| 分类 | Overlap / Scheduler(major) |
| Label | ready-to-merge |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/8144 |

## 一句话结论

TBO 在序列 token 分布失衡时改用 two-chunk-overlap(按 token 对半切长序列),DeepSeek-V3 prefill 吞吐最高 +12.56%。

## 动机

动机在关联 PR #6328 中提出:现有 two-batch-overlap(TBO)对 EXTEND batch 只能按"序列个数"二分(`_split_array_by_half_sum` 找让两边 token 和最接近的切分点),当 batch 里有一条超长请求(如单 DP 上 3072 token 的请求)时,该序列无法被拆开,两个 micro-batch 的 token 量严重失衡,注意力/MoE 的 overlap 收益被长序列独占拖垮。

## 优化点

引入 two-chunk-overlap:新增 `--tbo-token-distribution-threshold`(默认 0.48),`_is_two_chunk_split_enabled` 检测按序列均衡切分后左半 token 占比是否落在 [threshold, 1-threshold] 区间之外;失衡时改用 `_split_array_by_cum_less_than_half` 把整包 token 按累计和越过一半的位置切分,允许同一条长序列跨到两个 micro-batch(chunked prefill 式),`compute_split_token_index` 相应返回 `sum//2`。新增 `derive_fields_related_to_seq_len_for_two_chunk` 对两个 child ForwardBatch 做字段修正:child_a 末条序列长度截断为左半剩余 token,child_b 首条序列的 prefix len 加上左半尾巴,再用 `compute_position` 重算 `extend_start_loc`,并同步更新 device 侧 len/sum 字段;split 的 `end_seq_index` 加 1 以覆盖被拆开的序列。附带把 `forward_batch_info` 里按 backend 选 `compute_position_triton/torch` 的逻辑收敛成统一的 `compute_position`。

## 关键改动

```text
python/sglang/srt/two_batch_overlap.py: 失衡检测 _is_two_chunk_split_enabled、按 token 切分、child batch 字段派生
python/sglang/srt/server_args.py: 新增 --tbo-token-distribution-threshold(默认 0.48,0 表示禁用)
python/sglang/srt/model_executor/forward_batch_info.py: 抽出统一 compute_position
test/srt/test_two_batch_overlap.py: 单测改为同时校验 split_seq_index 与 split_token_index
```

```python
def _is_two_chunk_split_enabled(extend_lens):
    vanilla_split_seq_index = _split_array_by_balanced_sum(extend_lens)
    left_sum = sum(extend_lens[:vanilla_split_seq_index])
    overall_sum = sum(extend_lens)
    threshold = global_server_args_dict["tbo_token_distribution_threshold"]
    assert threshold <= 0.5
    return left_sum < overall_sum * threshold or left_sum > overall_sum * (1 - threshold)
```

## 性能声明

PR 正文,DeepSeek-V3-0324,2 节点 x 8 x H800,TP16/DP16,deepep normal + dp-attention + TBO 开启,disable-overlap-schedule/radix-cache/cuda-graph,`tbo-token-distribution-threshold 0.48`,bench_serving 每组跑 5 次丢第 1 次:

- 特例(每 DP 单条 3072 长度请求,input 3072/output 1/并发 1024):baseline 吞吐约 63900-64901 tok/s,本 PR 约 71511-73152 tok/s,平均 +12.56%。
- 一般例(变长输入 30-3072,2048 prompts,并发 1024):平均 +5.15%。

PR 注明提升幅度取决于输入长度分布;two-chunk-overlap 会触发 chunk prefill,实验还配合 `SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=1`。

## 验证与回归风险

`test/srt/test_two_batch_overlap.py` 的单测覆盖了 12 组 extend_lens 分布下 (seq_index, token_index) 的期望值,包括 `[99999,1,1,1,1]` 这类极端失衡(期望 two-chunk 切在 token 中点)与常规均衡用例;另有 `test_mmlu` 端到端精度用例。风险路径:two-chunk 分支改写了 child batch 的 `extend_seq_lens/prefix_lens/seq_lens/start_loc` 多个字段,依赖 `_update_device_and_sum_field_from_cpu_field` 的 CPU→device 同步正确性;阈值断言 `<= 0.5`,TBO+chunked prefill 与其他 prefill 特性(radix cache、overlap schedule 在实验中被关闭)的交互需注意;仅实验覆盖 DeepSeek-V3 形态,其他 MoE/DP-attention 模型未在正文中验证。

## 关联

动机与基准来自 #6328(TBO 原始工作)。pack 未列出其他同分类 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
