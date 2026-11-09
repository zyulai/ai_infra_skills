# PR #34184: Fix stale track rows corrupting conv checkpoints under the prefill graph

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34184](https://github.com/sgl-project/sglang/pull/34184) |
| 作者 | ispobock |
| 合并日期 | 2026-08-10 |
| labels | blackwell, run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
hybrid-SWA 模型 prefix-cache 命中时,replayed prefill graph 读到上一次重放残留的 `mamba_track_mask/indices` 行,track scatter 把当前窗口写进更早请求的 conv checkpoint;在 sentinel tail 清零中补上这两个缓冲。

## 问题现象
hybrid-SWA 模型上,prefix-cache 命中时可以解码到一个它从未产生过的 conv state。触发条件组合:prefill CUDA graph + chunked prefill + 并发 + prefix 命中(不需要 speculative decoding);单独禁用 prefill graph 即可消除。后果:之后任何恢复该 prefix 的请求,其整个生成过程都是错的。作者通过在插入时刻对 slot 做插桩,实测确认恰有两个 checkpoint 的 conv state 与不开 graph 的同次运行不同,且其声称的 track seqlens 正是两个出错请求的前缀——损坏是实测而非推断。

## 根因
正文披露:replayed prefill graph 会读取 `_capture_req_slots` 的所有行,但 `mamba_track_mask` / `mamba_track_indices` 保留了上一次重放越过当前 live batch 之后的残留值,导致 captured track scatter 把当前窗口落到更早请求的 checkpoint 里。代码中 sentinel tail 注释本就为此存在("the captured graph reads all req_slots entries, so stale values from the previous replay must be cleared")并清六个字段;但这两个 track 缓冲存放在 buffer registry 里,其清零由 `PrefillInputBuffers.fill_from` 中的 `bs != raw_bs` 守卫——prefill graph 按 token 而非请求数做 padding,传入 `padded_bs == raw_bs`,该分支永远不会执行,残留行从未被清。

## 修复方案
在 `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` 的 sentinel tail(清零 `extend_start_loc`/`req_pool_indices`/`orig_seq_lens` 等六字段处)补充:对 buffer registry 中的 `mamba_track_mask` 与 `mamba_track_indices` 两个 slot,把 `bs:r` 区间清零,与既有六个字段一起覆盖。

关键 diff 摘录:

```diff
+                # The captured track scatter reads these rows too, and a stale
+                # mask row still carries a live destination slot: it would land
+                # this replay's window in an earlier request's checkpoint.
+                registry = self.buffer_registry
+                for name in ("mamba_track_mask", "mamba_track_indices"):
+                    if registry.has_slot(name):
+                        registry.get_slot(name).buffer[bs:r].zero_()
```

另外,把 #34168 的 `test/registered/models_e2e/test_inkling_small_nvfp4.py` 中 steering 该 bug 的 `--disable-prefill-cuda-graph` 移除,让确定性 KL 用例(即该 bug 的回归守卫)回到 graph 路径。

作者还说明了备选方案(让 `padded_bs` 上报 captured slot 数,使 `fill_from` 的清零生效、避免死分支)更"诚实",但会改变共享这些缓冲的所有 phase 的行为,故本 PR 取局部修复。

A/B 验证(缩水 Inkling checkpoint,tp=1、32 并发、prefill graph 开、`--enable-deterministic-inference`,与 flushed reference prefill 对比 logprobs 差异数):prefill_cache_hit 512 tokens 1022/16384→0/16384,8 tokens 16/256→0/256。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 prefill CUDA graph runner 的 sentinel tail 清零逻辑:对 registry 中存在这两个 track slot 的模型(hybrid-SWA/mamba 类)多清零 padding 区间,非该类模型 `has_slot` 为假无操作;修复只在 `bs:r` 区间操作,不影响 live batch 行。CI 用例回到 prefill graph 路径后,该用例覆盖面扩大。作者提示:tp=4 真实 NVFP4 配置下 batch 打包不同,若 CI 仍红,说明同类缺口存在于别处,可用相同插桩定位。

## 关联
- https://github.com/sgl-project/sglang/pull/34043(spec 路径上相邻的失败:scatter 被 capture 特化掉;与本 bug 不同,本 bug 在含 #34043 时仍复现)
- https://github.com/sgl-project/sglang/pull/34168(引入 `test_inkling_small_nvfp4.py` 带 `--disable-prefill-cuda-graph` 的 PR,本 PR 把该 flag 移除)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
