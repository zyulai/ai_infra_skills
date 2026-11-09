# PR #44483: [Bugfix] Fix illegal memory access from a forward during a partial wake_up

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#44483](https://github.com/vllm-project/vllm/pull/44483) |
| 作者 | Meihan-chen |
| 合并日期 | 2026-06-24 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 13 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
sleep 模式下只唤醒 weights(KV 仍休眠)时调度器即恢复、dummy batch 照跑,触发 CUDA illegal memory access;改为 executor 完全苏醒(`is_sleeping` 为 False)后才恢复调度,dummy pass 在休眠期跳过。

## 问题现象
开启 `--enable-sleep-mode` 的 MoE 部署在部分唤醒后执行 forward 即崩溃。复现(4× A100-80GB,Qwen3-30B-A3B,TP2×DP2 + `--enable-expert-parallel --enable-sleep-mode`):`POST /sleep?level=1`(释放 KV)→ `POST /wake_up?tags=weights`(只唤醒权重,KV 仍休眠)→ `POST /v1/completions`。修复前该序列以 `CUDA illegal memory access` 崩溃(栈出自 `execute_dummy_batch` / `_compute_slot_mapping_kernel`,完整栈见 #44395);修复后请求挂起不崩溃,engine 保持暂停,待 `wake_up?tags=kv_cache` 后该 pending 请求立即返回 200 OK。全量唤醒(full wake)路径行为不变,回归测试通过(completion 正常返回)。

## 根因
PR 正文披露:修复原则是「只有 executor 完全驻留后才恢复调度」。`model_executor.is_sleeping` 在所有 sleep tags 唤醒前保持 True(`Executor.wake_up` 仅在 `sleeping_tags` 为空时清除它),但此前的 `wake_up` 无条件调用 `resume_scheduler()`,导致 weights-only 部分唤醒后调度器即恢复;此时 KV cache 尚未驻留,`run_busy_loop` 中的 dummy pass 照常执行,forward 访问了未驻留的显存,产生 illegal memory access。

## 修复方案
修改 `vllm/v1/engine/core.py` 两处:

1. `wake_up()`:仅当 `not self.model_executor.is_sleeping`(所有 tags 均已唤醒)时才 `resume_scheduler()`;部分唤醒有意让剩余分配保持休眠;
2. `run_busy_loop()`:`execute_dummy_batch()` 增加 `self.is_sleeping()` 与 `self.model_executor.is_sleeping` 两级门控,休眠期(含 KV-offload 窗口)不执行 dummy pass。

关键 diff 摘录:

```diff
-        # Resume scheduling (applies to all levels)
-        self.resume_scheduler()
+        # Partial wakes intentionally keep the remaining allocations asleep.
+        # Resume scheduling only once all executor memory is resident again.
+        if not self.model_executor.is_sleeping:
+            self.resume_scheduler()
```

## 影响与回归风险
- 影响窗口:正文未说明受影响的版本/提交范围。
- 回归面:仅触及 `core.py` 的 wake_up 与 busy loop 路径,影响所有使用 sleep mode / tag 唤醒的部署:部分唤醒后调度保持暂停,新请求会排队(表现为请求等待而非报错),直到剩余 tags 唤醒;全量唤醒路径行为无变化(正文声明并经回归测试验证)。若用户依赖部分唤醒后立即推理,行为会从崩溃变为挂起等待,属预期改变。不使用 sleep mode 的部署不受影响。

## 关联
- Fixes #44395(含崩溃栈): https://github.com/vllm-project/vllm/issues/44395

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
