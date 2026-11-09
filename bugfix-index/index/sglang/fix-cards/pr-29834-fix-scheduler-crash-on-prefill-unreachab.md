# PR #29834: Fix scheduler crash on prefill-unreachable decode abort

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29834](https://github.com/sgl-project/sglang/pull/29834) |
| 作者 | amote-i |
| 合并日期 | 2026-07-08 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 17 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
PD 分离下 prefill bootstrap 不可达时 abort 只清理 queue、留下 pending_reqs 悬挂引用导致下一次解析崩溃;失败请求改为按 id() 从两处一并删除,并给 abort 加 None 保护。

## 问题现象
当 prefill bootstrap server 不可达时,一个同时存在于 `self.queue` 与 `self.pending_reqs`(经 `add()` slow path 加入)的 DecodeRequest,其 `kv_receiver` 会被 abort 路径(PR #28022)置为 None,但只清理了 `self.queue`;`pending_reqs` 中残留的悬挂引用使下一次 `_resolve_pending_reqs` 崩溃:

```
AttributeError: 'NoneType' object has no attribute 'abort'
```

## 根因
正文披露:abort 路径(PR #28022)把请求的 `kv_receiver` 设为 None,但只从 `self.queue` 清理;`self.pending_reqs` 中同一 DecodeRequest 对象的引用未被清理,后续 `_resolve_pending_reqs` 对该引用调用 `.abort()` 时对 None 调用方法而崩溃。

## 修复方案
改 `python/sglang/srt/disaggregation/decode.py` 两处:

1. `pop_preallocated`:abort 循环收集 `failed_reqs` 后,按 **id()** 身份比较(因为 DecodeRequest 的 dataclass `__eq__` 会比较 tensor receiver)把它们从 `pending_reqs` 一并删除:

```python
if failed_reqs:
    failed_ids = {id(r) for r in failed_reqs}
    self.pending_reqs = [
        r for r in self.pending_reqs if id(r) not in failed_ids
    ]
```

2. `_ensure_prefill_info`:给 `.abort()` 加 backstop 保护——`kv_receiver` 可能已被先前的 self.queue 清理置为 None:

```python
if decode_req.kv_receiver is not None:
    decode_req.kv_receiver.abort()
```

另新增两个回归测试(`test_prealloc_abort_also_drops_from_pending_reqs` 用会抛 TypeError 的 `BadEqReceiver` 钉死身份比较语义、`test_ensure_prefill_info_tolerates_cleared_receiver`);正文指出现有测试套件把 `pending_reqs=[]` 写死,掩盖了这个 case。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:PD disaggregation decode 侧 `DecodePreallocQueue` 的失败请求清理路径;`pending_reqs` 只在存在失败请求时被按身份过滤,正常请求不受影响;None 保护仅在 receiver 已清理时跳过 abort,行为上只是不再崩溃。

## 关联
- PR #28022(正文文字提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
