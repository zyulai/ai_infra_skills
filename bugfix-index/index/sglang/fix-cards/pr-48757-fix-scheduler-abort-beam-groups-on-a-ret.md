# PR #48757: [Fix] scheduler: Abort beam groups on a retract pause and close three request-drop gaps

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#48757](https://github.com/sgl-project/sglang/pull/48757) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-11-06 |
| labels | run-ci, apple-silicon, bypass-fail-fast |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Overlap / Scheduler |
| bug 类型 | crash |

## 一句话结论
修四个请求状态漏洞:retract 暂停中止 beam 组、busy 时 release 不再崩溃、HiSparse staged 可被 abort、PD 切换拒绝持有请求。

## 问题现象
正文描述四类场景会把请求留在坏状态:
1. `pause_generation(mode="retract")` 只 requeue beam 组的 leader 而不是中止整个组;恢复后第一次 `filter_batch` 会触发 `reqs changed while a beam tail was attached`(陈旧 beam tail 未随请求清掉);无法恢复的 pause-retracted beam leader 还会泄漏 member rows。
2. scheduler 忙时调用 `release_memory_occupation`:idle 检查是裸 `assert`,会直接崩掉 scheduler,且在 `python -O` 下检查被跳过。
3. HiSparse 请求 prefill 后在 staging 列表等 host copy 落地、不属于任何 batch,此时到达的 abort 匹配不到任何东西,请求会一直跑完。
4. PD role switch 复用 `is_fully_idle`(它刻意跳过 retract 暂停持有的 `PD_HELD` 请求以允许 weight update flush cache),导致切换时拆掉 decode 队列、把被持有的请求不加应答地丢弃。

## 根因
正文 Changes 逐条披露:四个路径都缺少对应状态的收尾处理——retract 暂停未按 `ScheduleBatch.retract_decode` 的同款路径中止 beam 组并释放 member rows、`release_memory_occupation` 用断言代替可拒绝的检查、abort 的 in-flight 收集遗漏 HiSparse staging 列表、PD 切换误用跳过 held 请求的 idle 判定。

## 修复方案
- `schedule_batch.py`:`retract_all` 重写为对 beam leader 记录 `FINISH_ABORT`(503)、通过新增 `free_beam_member_rows` 先释放 member rows 再 release;不可恢复的请求同样记录 abort reason;返回 `reqs_to_abort`。`retract_decode` 中两处释放 member rows 收敛到同一函数。
- `scheduler.py`:`pause_generation` 改用 `running_batch.filter_batch(keep_indices=[])` 清空 running batch 使 beam tail 随请求剥离;`collect_inflight_reqs` 纳入 `HISPARSE_STAGING` 列表中的请求。
- `weight_updater.py`:`release_memory_occupation` 的裸 assert 改为返回 `ReleaseMemoryOccupationReqOutput(success=False, message=...)`;成功时返回 `success=True`。
- `io_struct.py`:`ReleaseMemoryOccupationReqOutput` 增加 `success`/`message` 字段;`tokenizer_control_mixin.py` 合并结果并在失败时抛 `RuntimeError`(HTTP 端点经既有错误路径变 400)。
- `role_switch.py`:存在 `PD_HELD` 持有请求时以 `safe_to_restore=True` 拒绝切换。
- 新增三个「修复前失败、修复后通过」的单测(beam 组 retract 中止、HiSparse staged 请求下一步结束、retract 暂停持有请求时拒绝 PD 切换)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:行为变化明确列出——retract 暂停期间 beam-search 请求以 503 中止且暂停后不恢复;忙时 `release_memory_occupation` 不再崩 scheduler,改为 `Engine`/`TokenizerManager` 抛 `RuntimeError`、HTTP 返回 400(成功路径返回值不变);HiSparse staging 中的 abort 现在会终结请求;持有请求时 PD role switch 被拒绝直到恢复并排空。受影响路径:beam search、pause/retract、weight updater/flush cache、HiSparse abort、PD role switch、MLX scheduler mixin(其 batch mock 补了 `filter_batch`)。

## 关联
正文未列出(未附 issue/PR 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
