# PR #16067: [bugfix] fix mamba slot leak when scheduling fails with radix cache (#15840)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16067](https://github.com/sgl-project/sglang/pull/16067) |
| 作者 | kuafou |
| 合并日期 | 2026-02-13 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 16 |
| 分类 | Overlap / Scheduler |
| bug 类型 | resource |

## 一句话结论
radix cache + Mamba 下 add_one_req 失败时,COW 预分配的 mamba 槽无人释放,check_memory 误报泄漏令服务崩溃;调度失败时显式释放修复。

## 问题现象
开启 `radix-cache` 服务 Mamba 模型时,高负载/请求回撤期间调度器暂时资源阻塞,服务崩溃并报 `ValueError: token_to_kv_pool_allocator memory leak detected!`,错误日志含 `leaked_mamba_pages={191}`、`mamba_available_size=315`、`mamba_evictable_size=138`、`mamba_pool.size=454` 等。问题在调查 #15840 时发现,但属于调度器失败路径清理的独立问题。

## 根因
正文披露的四步链条:
1. `init_next_round_input()` 里经 COW/`match_prefix` 分配了 Mamba 槽;
2. 随后 `add_one_req()` 返回 `NO_TOKEN`(如 KV cache pool 满);
3. 调度器进入 Idle 状态(此处实为"资源阻塞"),触发 `self_check_during_idle()` → `check_memory()`;
4. `check_memory()` 失败:该槽不在 `free_slots`(已分配),也尚未被 `tree_cache` 跟踪(请求未完成调度),被误判为泄漏。

## 修复方案
`scheduler.py` 的 `_get_new_batch_prefill_raw` 在 `res != AddReqResult.CONTINUE` 时调用新增的 `maybe_release_mamba_cache(req)`:若 `req.mamba_pool_idx` 非 None 且非 disaggregation DECODE 模式,则 `mamba_pool.free(...)` 并置 None;请求留在 waiting_queue,下一轮调度经 `match_prefix()` 安全地重新分配(并复制)mamba 状态。

```diff
             if res != AddReqResult.CONTINUE:
+                self.maybe_release_mamba_cache(req)
+
                 if res == AddReqResult.NO_TOKEN:
```

注意:disaggregation DECODE 模式下 mamba 状态来自 PREFILL 传输、释放后不可恢复,故不在此释放;为避免该场景误报,self_check_during_idle 在 waiting queue 非空时跳过内存检查。新增回归测试 `test_mamba_slot_release_after_match_prefix_cow`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:Mamba 模型 + radix cache 的调度失败路径(NO_TOKEN/IDLE 分支);disaggregation DECODE 模式保持不释放;idle 自检在 waiting queue 非空时跳过内存检查(以推迟检查换取消除误报)。纯资源清理,无模型输出变化。

## 关联
- 相关 issue:#15840

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
