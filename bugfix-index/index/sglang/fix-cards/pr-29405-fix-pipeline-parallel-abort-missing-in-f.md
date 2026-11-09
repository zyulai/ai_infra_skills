# PR #29405: Fix pipeline-parallel abort missing in-flight requests in non-current microbatch slots

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29405](https://github.com/sgl-project/sglang/pull/29405) |
| 作者 | fzyzcjy |
| 合并日期 | 2026-07-10 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
PP 下 abort 只扫描当前及上一步的 microbatch slot,其他 slot 的在跑请求收不到 `FINISH_ABORT` 继续占着 KV;改为 PP 下扫描全部 `running_mbs`/`mbs` slot 并去重。

## 问题现象
在 pipeline parallelism(PP)下执行 `abort_request`(含 `abort_all`)时,"abort 已在运行请求" 的路径漏掉请求:只有当前 slot 与上一个 slot 中的请求被标记 `FINISH_ABORT`,处于其他 microbatch slot 的 `running_mbs`(或仍在 `mbs` 中在途)的请求永远不会被 abort,继续解码到自然结束并一直持有 KV。触发条件:PP 且 microbatch slot 数大于 2——只有两个 slot 时"当前 + 上一步"恰好覆盖全部 slot,问题不显现。

## 根因
正文披露:`abort_request` 的"abort 已运行请求"路径只扫描 `self.running_batch` 加一个过期的 `self.cur_batch`。PP 下调度器把每个 microbatch slot 的状态保存在 `running_mbs[]` / `mbs[]` 中,abort 时刻 `self.running_batch` 只是别名指向*当前* microbatch slot,而 `self.cur_batch` 持有的是*上一步*的 batch。因此 abort 只能触达当前与上一个 slot,其余 slot 的请求漏网。

## 修复方案
改 `python/sglang/srt/managers/scheduler.py` 的 `abort_request`:PP 下改为扫描每个 `running_mbs` 与 `mbs` slot(按 request 去重);非 PP(`pp_size == 1`)保持原扫描 `running_batch` + `cur_batch` 的行为:

```diff
-        if self.cur_batch is self.running_batch or self.cur_batch is None:
-            reqs = self.running_batch.reqs
+        if self.ps.pp_size == 1:
+            inflight_batches = [self.running_batch, self.cur_batch]
         else:
-            reqs = self.running_batch.reqs + self.cur_batch.reqs
+            inflight_batches = [*self.running_mbs, *self.mbs]
 
-        for req in reqs:
+        inflight_reqs = {r for b in inflight_batches if b is not None for r in b.reqs}
+        for req in inflight_reqs:
```

并新增 scripted-runtime 回归测试 `test/manual/scheduler/test_scripted_pp_abort.py`(`pp_size=4`、`pp_max_micro_batch_size=1`):把 4 个 decode 请求各推进到不同的 microbatch slot,断言 `abort_all` 后全部释放 KV;正文说明无修复时该测试失败(非当前 slot 的请求仍持有 KV),修复后通过,RED/GREEN 证据见评论区。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`abort_request` 的在跑请求扫描逻辑。非 PP 路径行为保持;PP 路径扫描范围扩大到所有 slot,abort 命中的请求数量增加(这正是修复意图),依赖旧行为(abort 只影响当前 slot)的使用方会观察到更多请求被终止。新增测试文件仅用于手动/回归验证。

## 关联
- sgl-project/sglang#21300(此前调查尝试,未覆盖其他 slot 且无 scripted-runtime 测试)
- sgl-project/sglang#28881(另一尝试,捆绑了无关修复;本 PR 将 slot 扫描统一为单个去重循环)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
