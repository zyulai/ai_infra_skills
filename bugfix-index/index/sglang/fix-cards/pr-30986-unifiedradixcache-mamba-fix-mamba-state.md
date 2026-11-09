# PR #30986: [UnifiedRadixCache][mamba] Fix mamba state corruption and slot leak when load_back aborts

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30986](https://github.com/sgl-project/sglang/pull/30986) |
| 作者 | Jialin |
| 合并日期 | 2026-07-23 |
| labels | run-ci |
| 热度 | 👍 2 · 💬 17 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
mamba load_back 中止时不回滚已发布的 CoW 槽位,请求从未初始化/过期状态解码产生静默错误生成并泄漏槽位;改为 prepare/finalize 钩子在 finally 中回收。

## 问题现象
被中止的 mamba load-back 会静默破坏该请求的输出。触发条件:主机内存压力加中止时机(内存配额不足、驱逐不足、或 `cache_controller.load()` 返回 `None`)。症状是请求继续执行并解码,读取一个 H→D 拷贝从未运行的槽位——即未初始化内存或上一个占用者的过期递归状态,导致静默的错误生成。次要影响:槽位保持分配直到请求结束,构成有界泄漏;正文明确「corruption 才是严重的一半」。

## 根因
PR 正文披露:mamba LOAD_BACK transfer build 在 load-back 是否真正进行尚不可知之前,就分配 CoW device 槽位并发布到 `req.mamba_pool_idx`。当 `load_back` 中止(内存配额、驱逐不足、或 `cache_controller.load()` 返回 `None`)时,没有任何代码回退该动作:请求带着指向无效槽位的引用继续解码。由 diff 推断:具体机制是分配与发布发生在 `build_hicache_transfers` 内部,而中止分支(load 返回 None、配额检查失败)直接返回 False,不清理 `req.mamba_pool_idx` 与已分配槽位。

## 修复方案
把组件特定状态处理移入 `TreeComponent` 钩子(按 review 反馈):
1. `prepare_build_hicache_transfers(node, phase, *, req) -> PrepareTransfersResult`:在 transfer build 之前运行;mamba 实现分配并发布 CoW 槽位(含驱逐重试),记入结果的 `allocated_mamba_slot`;transfer build 不再分配,只断言槽位存在。
2. `postprocess_build_hicache_transfers(req, prep, success)`:mamba 实现在失败时释放槽位**并清空 `req.mamba_pool_idx`**,强制后续路径走合法的状态初始化。(diff 中对应实现为 `prepare_load_back`/`finalize_load_back` 钩子与 `PrepareLoadBackResult` 数据类。)
3. `load_back` 按 `ComponentType` 存储 prepare 结果并在 `finally` 中调用一次 finalize,使所有退出路径——中止分支、成功路径、意外异常——在单一位置结算 prepared 状态:

```python
preps = {comp.component_type: comp.prepare_load_back(...) for comp in ...}
success = False
try:
    success = self._load_back_transfers(...)
    return success
finally:
    for comp in self._components_tuple:
        comp.finalize_load_back(req, preps[comp.component_type], success)
```

非 mamba 组件为 no-op 钩子。部分成功说明:`cache_controller.load()` 是单次复合调用(KV + 全部 aux pool)返回 indices-or-None,该层无法观测 per-component 成功与否,全局 `success` flag 与 API 匹配。新增 3 个 Full+Mamba 测试,均验证过无修复时为红:quota 中止与 `load()` 失败都回收槽位并清空请求引用;调用前已拥有槽位的请求不受影响。全量单测绿(`test_unified_radix_cache_unittest.py`:850 passed / 702 skipped)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:改动集中在 `unified_radix_cache.py` 的 `load_back` 路径(拆出 `_load_back_transfers`)、`mamba_component.py` 的 HiCache 钩子、`tree_component.py` 新增默认 no-op 钩子与 `PrepareLoadBackResult`。非 mamba 组件钩子为 no-op,分配工作量不变仅位置移动;mamba load-back 失败路径新增槽位回收;成功路径行为不变(新槽位保持发布)。`build_hicache_transfers` 中原分配逻辑改为 `assert req.mamba_pool_idx is not None`,依赖 prepare 先行执行的调用顺序。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
