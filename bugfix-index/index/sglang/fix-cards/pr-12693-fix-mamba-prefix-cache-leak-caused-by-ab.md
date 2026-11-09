# PR #12693: fix  mamba prefix cache leak caused by abort

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#12693](https://github.com/sgl-project/sglang/pull/12693) |
| 作者 | yizhang2077 |
| 合并日期 | 2025-11-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 2 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
mamba radix cache 匹配后、alloc 前请求被 abort 会导致 mamba 状态内存泄漏;在 abort 流程补释放路径修复。

## 问题现象
在 mamba prefix cache 匹配场景下,若请求在特定窗口内被取消(abort),会偶发内存泄漏。正文给出的复现方式:启动 `python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --chunked-prefill-size 2048`,然后运行 `python3 -m sglang.bench_serving --backend sglang --dataset-name generated-shared-prefix --gsp-num-groups 50 --gsp-prompts-per-group 100 --gsp-system-prompt-len 10240 --gsp-question-len 1024 --gsp-output-len 128 --max-concurrency 50`,过一会儿取消该命令,内存泄漏会偶发出现。

## 根因
正文披露:在 mamba prefix cache match 中,需要 fork 一块空间把 mamba state 从 radix tree 拷贝出来,并在 `HybridReqToTokenPool.alloc` 被调用后把所有权移交给请求。如果请求在 `MambaRadixCache.match_prefix` 与 `HybridReqToTokenPool.alloc` 之间被 abort,这份已拷贝/已占用的 mamba state 无人释放,造成内存泄漏。

## 修复方案
1. `python/sglang/srt/managers/scheduler.py` 的 `abort_request`:对处于 queued 状态被 abort、且带 `mamba_pool_idx` 的请求,补调一次 `self.tree_cache.cache_finished_req(req, is_insert=False)` 触发释放。
2. `python/sglang/srt/mem_cache/mamba_radix_cache.py` 的 `cache_finished_req`:
   - 用 `cache_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)` 安全截取 token_ids,避免 output_ids 为空时越界;
   - mamba_value 直接取 `req.mamba_pool_idx` 而不再经 `get_mamba_indices(req.req_pool_idx)`(abort 场景下 req_pool_idx 可能尚未分配);
```diff
-        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
+        cache_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
+        token_ids = (req.origin_input_ids + req.output_ids)[:cache_len]
...
-        mamba_value = (
-            self.req_to_token_pool.get_mamba_indices(req.req_pool_idx)
-            .unsqueeze(-1)
-            .clone()
-        )
+        mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
```
   - 释放逻辑分支:`req.req_pool_idx is not None` 时照旧 `free(req_pool_idx, free_mamba_cache=mamba_exist)` 并 `dec_lock_ref`;否则(abort 情况)直接 `self.req_to_token_pool.mamba_pool.free(mamba_value)` 释放 mamba 状态。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动集中在使用 mamba radix cache 的混合架构模型(如 Qwen3-Next)的请求完成/取消路径:`cache_finished_req` 的 token 截断、mamba 索引来源与释放分支,以及 scheduler 对 queued abort 请求的额外 cache_finished_req 调用;若 abort 时请求已部分执行,`is_insert=False` 保证不向 radix tree 插入新条目,但该路径上 `dec_lock_ref`/free 的时序变化需依赖 `req_pool_idx` 判空正确。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
