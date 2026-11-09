# PR #41927: [PD] Fix crash for allocation when mamba + pd + decode radix cache

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#41927](https://github.com/sgl-project/sglang/pull/41927) |
| 作者 | ByronHsu |
| 合并日期 | 2026-10-06 |
| labels | documentation, quant, amd, dependencies, lora, deepseek, speculative-decoding, hicache, sgl-kernel, npu, run-ci, mthreads, jit-kernel, memory-pool |
| 热度 | 👍 0 · 💬 18 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
Mamba+PD+decode radix cache 下准入只判零空闲槽,漏算 2 个跟踪缓冲致断言崩溃;改为先确保容量、不足即排队。

## 问题现象
Mamba/KDA 模型 + PD disaggregation decode + decode radix cache + overlap scheduling 场景下,decode worker 在预分配时崩溃:`AssertionError: Not enough space for mamba ping pong idx`,调用链为 `DecodePreallocQueue.pop_preallocated -> _pre_alloc (decode.py:2075) -> HybridReqToTokenPool.alloc (memory_pool.py:1471) -> _alloc_ping_pong_buffer`。作者在单 H200 节点(两 GPU)用 tiny Kimi Linear 真实权重、NIXL PD、16 个顺序独立请求(每条 257 输入 / 256 输出 token)复现:5 个 Mamba 槽位时前 3 条请求完成后,第 4 条在「2 空闲 + 3 可逐出槽位」处崩溃;128 槽位对照组全部完成。

## 根因
正文披露:main 分支的 PD 准入(admission)检查只判断 Mamba 空闲槽数是否为 **0**,即使还剩 1 个空闲槽也不逐出缓存状态;而开启 overlap scheduling 时一个新请求实际需要 **3** 个槽位——1 个活跃状态加 2 个跟踪缓冲(tracking buffers)。于是准入放行后,请求行与活跃 Mamba 状态先分配(空闲 1→0),两个跟踪缓冲无法再分配,`_alloc_ping_pong_buffer` 断言失败并使 decode worker 退出。

## 修复方案
- 从 combined-engine 的分配逻辑中抽出 `ensure_mamba_capacity`(allocation.py):按每个请求仍需的活跃状态与跟踪槽位计数,不足时经 `tree_cache.evict_for_alloc` 逐出足够缓存状态再复查,返回容量是否满足。
- PD decode 侧(decode.py)在 `pop_preallocated` 预分配前与 `resume_retracted_reqs` 恢复被撤回请求前调用该检查,仍不足则让请求留在队列、不做部分分配;同时以 `allocation_sizing.get_mamba_tracking_slots` 统一跟踪槽位计算,`HybridReqToTokenPool._alloc_ping_pong_buffer` 改用 `mamba_initial_tracking_slots`。关键改动:
```python
if not ensure_mamba_capacity(
    self.req_to_token_pool, [decode_req.req], self.tree_cache
):
    break
```
- combined serving 的 `alloc_req_slots` 保持原行为:先尝试逐出再交给分配器(原 `MAMBA_STATE_PER_REQ_*` 常量与 3x/1x factor 逻辑被该函数取代)。
- 验证:修复后 5 槽位 16 条请求全部完成,重复 prompt 命中 256 cached tokens;单测 13 passed, 3 subtests passed(H200)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:PD decode 预分配与撤回恢复路径新增容量检查(容量不足会延迟而非崩溃);combined serving 的 `alloc_req_slots` Mamba 预检逻辑重写;Mamba 池容量计算(`slots_per_req` 改按 `mamba_initial_tracking_slots`)与 `kv_cache_configurator` 的 mamba ratio 计算改用新函数,overlap/lazy 模式跟踪槽位定义随之变化。非 hybrid 模型经 `isinstance(HybridReqToTokenPool)` 检查直接放行,不受影响。

## 关联
正文未列出 issue/PR 链接(仅引用 main 分支代码行位置)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
