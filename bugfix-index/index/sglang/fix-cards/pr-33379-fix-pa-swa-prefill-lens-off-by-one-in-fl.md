# PR #33379: Fix _pa_swa_prefill_lens off-by-one in FlashAttentionBackend

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#33379](https://github.com/sgl-project/sglang/pull/33379) |
| 作者 | DmiSergeev |
| 合并日期 | 2026-08-07 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

`_pa_swa_prefill_lens` 按 req_pool_idx 直接索引却只分配 size 个元素,差一越界;改为 size+1 修复 CI 设备断言崩溃。

## 问题现象

在 `base-b-test-1-gpu-large` CI 的 `test_unlimited_ocr_server.py` 中,decode CUDA-graph capture 刚结束、第一个真实请求到达时,CUDA `index_put` 抛出 "index out of bounds" 设备断言(作者附 CI run 链接)。触发条件:模型启用 `is_prefill_aware_swa()`(如 `python/sglang/srt/models/unlimited_ocr.py`),且有请求落在 `req_pool_idx == size` 上。

## 根因

正文披露:`FlashAttentionBackend.__init__` 为 prefill-aware SWA 分配的暂存缓冲 `_pa_swa_prefill_lens` 尺寸取 `max_bs = model_runner.req_to_token_pool.size`,但它不是按 batch 位置索引,而是直接用原始 `req_pool_idx` 索引(写入与 `_build_pa_page_table` Triton kernel 内的读取都是如此)。`ReqToTokenPool` 把 row 0 保留为 CUDA-graph padding 槽,实际派发的槽位是 `1..size`(`_alloc_size = size + 1`),因此合法 `req_pool_idx` 范围是 `[0, size]` 共 size+1 个值;缓冲只开 size 个即差一,`req_pool_idx == size` 的请求会越界。

该 bug 长期潜伏,因为旧的 `ReqToTokenPool.alloc()` 从 `free_slots` 头部升序取槽,`size` 只有在池几乎满载时才会被派发,普通 CI 跑不到;#32208 改为从尾部降序取槽(全新池的第一次分配就拿到 `req_pool_idx == size`),才把它暴露出来。作者还 grep 确认 `python/sglang/srt` 下按池槽 id 索引的缓冲仅此一处,其余 `max_bs` 尺寸缓冲(kv_indptr 等)均按 `[:bs]` 位置索引,不受影响。

## 修复方案

`python/sglang/srt/layers/attention/flashattention_backend.py` 的 `__init__` 中,把缓冲尺寸改为 `req_to_token_pool.size + 1`,与 `ReqToTokenPool` 自身分配(`req_to_token.shape[0] == size + 1`)对齐,使 `[0, size]` 全部可寻址:

```diff
-            max_bs = model_runner.req_to_token_pool.size
+            max_req_pool_idx = model_runner.req_to_token_pool.size
             self._pa_swa_prefill_lens = torch.zeros(
-                max_bs, dtype=torch.int32, device=model_runner.device
+                max_req_pool_idx + 1, dtype=torch.int32, device=model_runner.device
             )
```

作者强调这是修真正的越界 bug,而不是把分配器改回低索引优先来再次掩盖问题。新增回归测试 `test/registered/unit/layers/attention/test_flashattention_pa_swa_prefill_lens_size.py`(注册于 `base-b-test-1-gpu-small`),断言缓冲 shape 为 `size + 1` 且在 `req_pool_idx == size` 写入不越界;本地 CPU 验证:修前 `IndexError: index 8 is out of bounds for dimension 0 with size 8`,修后通过(pool_size=8 时缓冲 shape 为 9)。

## 影响与回归风险

- 回归面:仅改变该预分配暂存缓冲的大小,不改动任何张量值、KV cache 内容或模型输出;受影响路径限于启用 prefill-aware SWA 的 FlashAttentionBackend(缓冲分配、写入与 `_build_pa_page_table` 读取)。

## 关联

- #32208(O(1) slot allocation in `ReqToTokenPool.alloc()`,使 bug 显性化)
- CI 失败 run:[base-b-test-1-gpu-large job](https://github.com/sgl-project/sglang/actions/runs/30669370441/job/91419918042)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
