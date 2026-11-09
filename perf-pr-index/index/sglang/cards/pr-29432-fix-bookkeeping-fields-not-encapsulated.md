# PR #29432: Fix bookkeeping fields not encapsulated with real allocations in normal alloc, PD pre-alloc, DFlash and EAGLE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @fzyzcjy |
| 合并 | 2026-07-15 |
| 分类 | Speculative Decoding |
| Label | bypass-fastfail |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 10 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29432 |

## 一句话结论

把 `req.kv.kv_allocated_len` 记账与真实 KV 分配封装进同一个分配函数(normal alloc、PD pre-alloc、DFlash、EAGLE 四条路径),消除远端 glue site 的簿记不一致风险。

## 动机

此前 `kv_allocated_len` 等记账字段散落在离真实分配很远的位置:PD pre-alloc 的 `_pre_alloc` 里 pool 分配与 `req.kv` 更新分离(代码中留有 TODO:应合并为单一 owned-kv 分配步骤)、`prepare_for_decode`/`prepare_for_extend` 在调度侧逐请求自增、DFlash 与 EAGLE 各自维护一套 spec-decode 分配逻辑(含重复的 `ALLOC_EXTEND_FUNCS` dispatch 表)。这种"分配在一处、记账在另一处"的结构容易在新增路径时漏更新。本 PR 是 `req_pool_idx` / cache / owned-KV 解耦的 stacked refactor chain 的一环(stacked on op40,聚合 CI sandbox #28636)。

## 优化点

核心是封装与统一。每条路径的物理分配函数在自己内部更新自己的簿记:PD pre-alloc 抽出 `alloc_for_decode_prealloc` / `alloc_for_decode_prealloc_hisparse`(留在 `decode.py`),内含 `req.kv` 创建/更新与 `swa_evicted_seqlen` 写入;spec-decode 侧把 DFlash 与 EAGLE 的分配统一为一个 `alloc_for_spec_decode`(移入 `allocation.py`,取消 `dsv4_npu_aware` 标志),内含 `alloc_token_slots`/paged extend/NPU dispatch、`assign_req_to_token_pool_func` 写 req_to_token 与 `kv_allocated_len` 高水位更新;`alloc_for_extend` / `alloc_for_decode` 也各自内联 `req.kv.kv_allocated_len` 更新。结构上把 `assign_req_to_token_pool` triton 内核从 `speculative/triton_ops/cache_locs.py` 迁到 `mem_cache/allocation.py`。一处**有意**的语义变化:DFlash 现在与 EAGLE 一样经同一 device dispatch 路由,在 DSV4 + Ascend NPU 上走 DSV4 reserve 路径(注:DFlash 从未发布过真实 NPU 路径,该角落原本是死的);CUDA 与非 DSV4 NPU 上行为等同。提交纪律:纯重定位为机器校验的 extract/move(gist 提供 verify_all.sh,PASS x4),语义重塑单独提交做等价审查。

## 关键改动

```text
python/sglang/srt/mem_cache/allocation.py: 新增 alloc_for_spec_decode + assign_req_to_token_pool 迁入 + ALLOC_EXTEND_FUNCS
python/sglang/srt/disaggregation/decode.py: 抽出 alloc_for_decode_prealloc[_hisparse],簿记内联
python/sglang/srt/speculative/eagle_utils.py: eagle_prepare_for_decode 改走 alloc_for_spec_decode
python/sglang/srt/speculative/dflash_info_v2.py: prepare_for_decode 改走 alloc_for_spec_decode
python/sglang/srt/managers/schedule_batch.py: prepare_for_extend/decode 移除远端记账
python/sglang/srt/mem_cache/common.py: release_kv_cache 拆出 _release_overallocated_kv_indices
```

## 性能声明

PR 未注明任何性能数字(重构定位,承诺行为等价;唯一语义变化是 DFlash 在 DSV4+NPU 上改走 DSV4 reserve 路径,原文说明该角落此前为死代码)。

## 验证与回归风险

机器校验:4 个 certified 重定位可从 base 字节级重现(gist `verify_all.sh` PASS x4);语义重塑提交逐个等价审查;聚合 CI sandbox #28636。单测更新:`test_decode_bookkeeping_ownership.py` 把 `kv_allocated_len` 的所有权位置从 schedule_batch/DFlash/eagle_prepare_for_decode 迁到 alloc_for_extend/alloc_for_decode/alloc_for_spec_decode/alloc_for_decode_prealloc[_hisparse](每个赋值点恰好 1 处的静态约定检查);hisparse allocator 单测改为断言 `req.kv.kv_allocated_len` / `req.kv.swa_evicted_seqlen`;radix cache 单测的 `_make_req` 改构造 `ReqKvInfo`,`cache_finished_req` 传入 `kv_len_to_handle`。风险面:CI Base/Extra 均失败(x,bypass-fastfail label);DFlash+NPU 行为变化虽然理论上无影响,但依赖"DFlash 从未有 NPU 用户"的前提;`release_kv_cache` 的 mamba/free 逻辑移入主函数尾部、over-alloc 释放拆出,边界顺序需保持。

## 关联

前序:op40 与 op28(owned-kv alloc 函数)、`req_pool_idx`/cache/owned-KV 解耦链;聚合 CI sandbox #28636。同区域:投机解码分配路径(EAGLE、DFlash)与 PD disaggregation pre-alloc、HiSparse allocator(PR #20343)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
