# PR #29432: Fix bookkeeping fields not encapsulated with real allocations in normal alloc, PD pre-alloc, DFlash and EAGLE

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29432](https://github.com/sgl-project/sglang/pull/29432) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-07-15 |
| labels | bypass-fastfail |
| 热度 | 👍 1 · 💬 30 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
把 req.kv.kv_allocated_len 等 bookkeeping 从相距很远的 glue 站点移进每条物理分配路径内部,统一 EAGLE/DFlash 的 spec-decode 分配入口 alloc_for_spec_decode。

## 问题现象
正文未描述具体运行时故障。该 PR 是 req_pool_idx / cache / owned-KV 解耦重构链的一部分,处理的是结构问题:normal alloc、PD pre-alloc、DFlash、EAGLE 四条路径上,`req.kv.kv_allocated_len` bookkeeping 未与真实物理分配封装在一起,而是在远处的 glue 站点各自更新。

## 根因
正文未披露某个具体 bug 的根因;按正文表述,问题在于 bookkeeping 与真实分配分离:每个物理分配本应在一个函数内更新自己的 bookkeeping,而不是由 far-away glue site 维护。

## 修复方案
- 把 `assign_req_to_token_pool` triton helper 从 `speculative/triton_ops/cache_locs.py` 搬到 `mem_cache/allocation.py`(certified move)。
- `disaggregation/decode.py`:抽出 `alloc_for_decode_prealloc` / `alloc_for_decode_prealloc_hisparse`,`_pre_alloc` 内的 `req.kv` 初始化与 `kv_allocated_len` 更新移入其中并删除原 glue 处更新。
- `allocation.py`:新增 `alloc_for_spec_decode`(含 `num_needed_tokens > 0` guard、设备分发表 `ALLOC_EXTEND_FUNCS`、`assign_req_to_token_pool_func` 调用与 `kv_allocated_len = max(...)` bookkeeping);`alloc_for_extend` / `alloc_for_decode` 内部更新 `kv_allocated_len`。
- `schedule_batch.py`:删除 `prepare_for_extend` / `prepare_for_decode` 中的 `kv_allocated_len` 更新;`eagle_utils.py`、`dflash_info_v2.py` 的分配代码统一路由到 `alloc_for_spec_decode`(EAGLE 侧 `ALLOC_EXTEND_FUNCS` 随之删除)。
- `mem_cache/common.py`:把释放超分配 KV 的逻辑抽成 `_release_overallocated_kv_indices`。
- 正文说明的一个 deliberate 变化:DSV4 + Ascend NPU 上 DFlash 现在像 EAGLE 一样走 DSV4 reserve 路径(DFlash 从未发布过真正的 NPU 路径,该角落是死代码)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:normal extend/decode、PD 预分配、DFlash、EAGLE 全部 KV 分配路径的 bookkeeping 时序与归属;`release_kv_cache` 被拆分;DSV4+NPU 上 DFlash 行为改变;测试从 `req.kv_allocated_len` 改为 `req.kv.kv_allocated_len` / `req.kv.swa_evicted_seqlen`,外部依赖旧字段的代码需同步。

## 关联
- 正文提及:Stacked on op40;Aggregated CI sandbox:#28636;验证脚本 gist:ec486009606de4216efe8359131da16c

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
