# PR #10846: Fix eagle radix cache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#10846](https://github.com/sgl-project/sglang/pull/10846) |
| 作者 | ispobock |
| 合并日期 | 2025-09-30 |
| labels | run-ci |
| 热度 | 👍 11 · 💬 3 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
EAGLE 开 radix cache 反而更慢、accept length 更低;radix 树改用 EAGLE bigram key 并修正插入/释放的长度差一,使缓存收益恢复正常。

## 问题现象
EAGLE3(Llama-3.1-8B-Instruct + `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`,`--speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --dtype float16`)下跑 `benchmark/mtbench/bench_sglang_eagle.py --parallel 1 --num-questions 10`:main 分支开启 radix cache 时吞吐 255.74 token/s、Acceptance length 2.26,反而低于关闭 radix cache 的 269.86 token/s、2.39——即缓存命中不仅没加速,还拖慢并降低接受率。

## 根因
正文未写明根因,仅给出上述对比数据。(由 diff 推断,diff 注释披露机制)EAGLE 的 draft 序列相对 target 是"整体左移一个 token",radix 树键需用 bigram 形式(`[1,2,3,4] -> [(1,2),(2,3),(3,4)]`),原 RadixCache 按 unigram key 插入/匹配,且 `cache_finished_req`/`cache_unfinished_req` 在 EAGLE 下按 token 数对齐 KV 长度,未考虑 bigram key 比 token 序列短 1,导致插入的 KV 索引错位、释放长度计算错误(`len(req.prefix_indices)` 与实际匹配长度不一致),命中结果不可用甚至引入额外开销。

## 修复方案
- `radix_cache.py`:RadixCache 新增 `is_eagle` 参数(scheduler 按_spec_algorithm 传入);`_convert_to_bigram_key` 把 token 序列转成相邻二元组 key,`match_prefix`/`insert` 先经 `key_convert_fn` 转换;insert 时 `value = value[: len(key)]` 对齐 bigram 长度。
- `cache_finished_req`/`cache_unfinished_req`:EAGLE 下 `actual_kv_len = all_token_len - 1`、`page_aligned_token_len = page_aligned_len + 1`,并用 `old_prefix_len`(考虑 partial page 与 EAGLE 多出的一个未匹配 token)替换原来的 `len(req.prefix_indices)` 计算释放区间。
- `schedule_batch.py`:Req 新增 `last_matched_prefix_len`,`init_next_round_input` 记录最近一次匹配前缀长度,用于后续正确释放 partial page 部分,避免内存泄漏。
- 新增单测 `test_insert_and_match_eagle`/`test_insert_and_match_eagle_page_size` 与 e2e `TestEAGLERadixCache`。

关键 diff(radix_cache.py):

```python
def _convert_to_bigram_key(tokens: List[int]) -> List[Tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:RadixCache 的 insert/match/cache_finished_req/cache_unfinished_req 主路径(非 EAGLE 时 `key_convert_fn` 为恒等,长度逻辑走原分支);Req 增加 `last_matched_prefix_len` 字段;正文兼容性自测:page size 16、page 16+chunked prefill 64、page 2/1+chunked prefill 64、单 batch 多 prompt(chunked prefill 角落场景,见 #10991);HiCache 未适配本修复但不破坏现有 HiCache(正文原话)。修复后 w/ radix cache 达 270.85 token/s、2.39,与 w/o 持平。

## 关联
- https://github.com/sgl-project/sglang/pull/10991

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
