# PR #9581: fix: revert #8593

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9581](https://github.com/sgl-project/sglang/pull/9581) |
| 作者 | zhyncs |
| 合并日期 | 2025-08-25 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
整体回滚 #8593:撤销 flashinfer MLA 后端 page_size!=1 支持,恢复 token 粒度 kv_indices 与 page_size=1 行为。

## 问题现象
正文未描述具体现象——Motivation、Modifications 等小节均为空模板。标题指明本 PR 是对 #8593 的回滚(fix: revert #8593),即 #8593 引入的改动造成了需要回滚的问题,但具体症状正文未给出。

## 根因
正文未披露根因。

(由 diff 推断)本 PR 把 flashinfer MLA 后端恢复到 #8593 之前的实现:`create_flashinfer_kv_indices_triton` 去掉 `PAGE_SIZE` 参数、回到 token 粒度写入;decode/prefill 的 `kv_indptr` 恢复为对 `paged_kernel_lens`(即 seq_len)做 cumsum,而非每请求页数;`wrapper.plan` 显式传 `page_size=1`;删除 `k_buf.view(-1, self.page_size, ...)` 页视图与相关换算;删除 `server_args` 中"投机解码 + page_size!=1 + flashinfer"的报错。即撤销 #8593 引入的 page_size!=1 多页支持。

## 修复方案
- `python/sglang/srt/layers/attention/utils.py`:`create_flashinfer_kv_indices_triton` 恢复 token 粒度实现(删除 PAGE_SIZE 参数、页换算与大段 docstring):

```diff
+    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
+    for i in range(num_loop):
+        # index into req_to_token_ptr needs to be int64
+        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
+        mask = offset < kv_end - kv_start
```

- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`:删除预分配的 `self.kv_indices`(恢复 cuda graph 下 `max_bs * max_context_len` 的临时分配)、`init_forward_metadata_replay_cuda_graph` 恢复 `cumsum(kv_len_arr_cpu)`、decode/prefill updater 恢复 `cumsum(paged_kernel_lens)` 与 `kv_indices = torch.empty(paged_kernel_lens_sum, ...)`、`plan` 恢复位置参数且 page_size=1、删除 import 来源改回 `utils`、删除 DP attention 下 `max_bs = dp_size * max_bs` 的 pre-init hack、`forward_extend/decode` 删除页视图。
- `python/sglang/srt/server_args.py`:删除 page_size!=1 + flashinfer 投机解码的 `ValueError` 校验。
- 测试:`test/srt/test_create_kvindices.py` 恢复无 page_size 维度的版本;`test/srt/test_mla_flashinfer.py` 删除 `TestFlashinferMLAPageSize16`。

## 影响与回归风险
- 影响窗口:自 #8593 合并起至本回滚。
- 回归面:flashinfer MLA 后端的 decode/prefill 元数据构建、cuda graph 捕获与重放、`forward_extend`/`forward_decode`;page_size!=1 的多页支持随之失效(`plan` 固定传 1,校验也已删除,设置该组合将回退到 #8593 之前的行为)(由 diff 推断);回滚后的代码含一处 `req_pool_indices: torch.Tnesor` 注解拼写(恢复自旧代码,diff 中可见)。投机解码 + flashinfer + page_size!=1 的报错不再出现。

## 关联
- 回滚对象:sgl-project/sglang#8593

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
