# PR #49328: [KV Offload] Fix failed-load livelock by marking the lookup verdict as a miss

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#49328](https://github.com/vllm-project/vllm/pull/49328) |
| 作者 | RobbieJ |
| 合并日期 | 2026-08-09 |
| labels | ready, v1 |
| 热度 | 👍 2 · 💬 28 |
| 分类 | KV Cache / Memory |
| bug 类型 | general |

## 一句话结论
二级层加载失败后 lookup 缓存仍报 HIT 令请求死循环;失败时 mark_miss 纠正裁决并收紧删文件条件。

## 问题现象
KV offload 中,secondary → primary 的 promotion 加载失败(块文件被截断、删除或不可读)后,请求被永久卡死:调度器每步都看到同样的 HIT、重新调度同一个注定失败的加载,形成空转 busy-loop,浪费 I/O 与调度,直到请求被 abort。由于 lookup 缓存按 block 引用计数,陈旧的 HIT 裁决会波及所有重叠这些块的请求——一个坏文件静默卡死多个请求。作者在 vLLM Metal 后端(Metal/MLX)上因 torn block file 复现,但问题不在 Metal:它位于共享分层层,纯 CPU 无 GPU 也能复现(有回归测试)。

## 根因
正文披露:异步 lookup 缓存记下正向(HIT)裁决后,加载失败的事实从不反馈给它;缓存被逐块引用计数共享,失败后依旧向调度器返回 HIT。次要根因:#49152 引入的"任何错误都删除块文件"会把瞬时故障变成永久数据丢失。

## 修复方案
1) 加载失败时, owning tier 在 `get_finished_jobs`(调度线程)对失败 keys 调用 `async_lookup.py` 新增的 `mark_miss`,把缓存裁决改为 False;缓存 miss 不再重新探测,请求直接在 GPU 重算,结构上无法再循环;`drain_results` 加 enqueue-once 断言防止迟到/重复结果复活旧裁决。2) `csrc/fs_io.cpp` 与 `vllm/v1/kv_offload/tiering/fs/io.py` 的 `load_block` 只在可证明的 short read(存储是原子的,过短即真损坏)时删文件,打开失败/读错误/满读后的 close 失败均不动文件。3) 批量加载失败时经 `num_succeeded` 上报已成功块,仅对失败尾部 mark_miss。

```diff
+    def mark_miss(self, keys: Collection[OffloadKey]) -> None:
+        for key in keys:
+            state = self._lookup_state.get(key)
+            if state is not None:
+                state.result = False
```

## 影响与回归风险
- 影响窗口:正文未给出具体提交/版本范围。
- 回归面:fs tier 与 obj tier 的 `get_finished_jobs`、`submit_load`,共享 `async_lookup` 与 fs I/O 层。行为变化:失败块当步即被判 MISS 并重算;瞬时读错误不再删文件(#49152 的 delete-on-any-error 被收窄);批量加载保留部分成功块。lookup 未改为 size 校验(评审后放弃,热路径更重且不可靠)。测试 342/352 passed,mutation-checked。

## 关联
- https://github.com/vllm-project/vllm/issues/49176
- PR #49152
- PR #56558
- https://github.com/vllm-project/DeepGEMM(pull)#16 — 正文以 "vllm-project/DeepGEMM#16" 提及

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
