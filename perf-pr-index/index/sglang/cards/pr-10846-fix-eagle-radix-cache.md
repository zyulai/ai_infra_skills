# PR #10846: Fix eagle radix cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ispobock |
| 合并 | 2025-09-30 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留 4 个核心文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/10846 |

## 一句话结论

修 EAGLE 下 radix cache 错配导致的负收益:bigram key 对齐后,开缓存吞吐从 255.74 恢复到 270.85 token/s,acceptance length 回到 2.39。

## 动机

EAGLE 的 draft 序列是 target 序列的 one-token-shifted 版本,radix tree 匹配必须用 bigram key(`[1,2,3,4] → [(1,2),(2,3),(3,4)]`)。此前实现未做该转换,导致开 radix cache 反而有害:实测(MTBench 10 题,parallel 1)EAGLE3 w/o cache 269.86 token/s、AL 2.39,而 w/ cache 掉到 255.74 token/s、AL 2.26——错配的 KV 复用破坏了 draft 验证语义,缓存放大为负收益。

## 优化点

(1) `RadixCache` 新增 `is_eagle` 标志(scheduler 初始化时按 `spec_algorithm.is_eagle()` 传入),`key_convert_fn` 在 EAGLE 下指向 `_convert_to_bigram_key`,`match_prefix`/`insert` 入口统一先转换 key;`insert` 时把 value 截断到 bigram key 长度。(2) `cache_finished_req`/`cache_unfinished_req` 重新推导边界:EAGLE 下 `actual_kv_len = all_token_len - 1`,插入用的 token 长度为 `page_aligned_len + 1`(bigram key 对应的 normal key 长度);page size 1 时也 free 掉多余的一个 KV 槽。(3) 新增 `Req.last_matched_prefix_len`(在 `init_next_round_input` 记录最近一次前缀匹配长度):page size > 1 时 partial page 会附加进 `prefix_indices` 但不入树,`old_prefix_len` 需要回退 1 个 token,才能在后续 `cache_unfinished_req`/`cache_finished_req` 中正确 free,避免内存泄漏。

## 关键改动

```text
python/sglang/srt/mem_cache/radix_cache.py: is_eagle + _convert_to_bigram_key + insert/match/finish/unfinished 边界修正
python/sglang/srt/managers/schedule_batch.py: Req.last_matched_prefix_len 记录
python/sglang/srt/managers/scheduler.py: RadixCache 构造传 is_eagle
```

## 性能声明

PR 正文(Llama-3.1-8B-Instruct + lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B,EAGLE3,steps 2 / topk 1 / draft tokens 3,fp16,`bench_sglang_eagle.py --parallel 1 --num-questions 10`,硬件未注明):

- main:w/o radix cache 269.86 token/s、AL 2.39;w/ radix cache 255.74 token/s、AL 2.26(负收益)
- 本 PR:w/o 269.45 token/s、AL 2.39;w/ 270.85 token/s、AL 2.39(与不开缓存持平略超)

## 验证与回归风险

作者列出兼容性自查:page size 16;page 16 + chunked prefill 64;page 1/2 + chunked prefill 64;HiCache(本修复未适配 HiCache,但不破坏现有行为);同 batch 多 prompt(chunked prefill corner case,ref #10991)。风险面:bigram key 改变树内 key 语义,所有 EAGLE 相关 cache 路径(match/insert/evict/free)都依赖新的长度约定;非 EAGLE 路径 `key_convert_fn` 为恒等,不受影响;page size > 1 的 partial-page free 逻辑依赖 `last_matched_prefix_len` 的时序正确性。

## 关联

同区域:同分类 Speculative Decoding 的 PR #3822(EAGLE small vocab)、PR #42297(block verification);ref #10991。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
