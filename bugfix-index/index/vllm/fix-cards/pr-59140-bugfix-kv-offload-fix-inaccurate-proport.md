# PR #59140: [Bugfix][KV Offload] Fix inaccurate proportional store dropping

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#59140](https://github.com/vllm-project/vllm/pull/59140) |
| 作者 | thuongvu |
| 合并日期 | 2026-10-14 |
| labels | bug, ready |
| 热度 | 👍 0 · 💬 14 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论

KV offload 的 `ThrottledDropPolicy` 用「每 N 次丢一次」近似比例丢弃,75% 实际只丢 50%;改为累加分数丢弃率,误差不超过 1 次。

## 问题现象

压力下按比例丢弃 store 请求时比例不准:以 75% 丢弃率、20 次 store 尝试为例,期望丢 15 次放行 5 次,实际只丢 10 次放行 10 次(50%)。该策略控制对二级缓存层的可选写入,不影响推理请求准入;压力检测与冷却行为不受影响。

## 根因

正文披露:旧实现把丢弃率取整为「每 N 次尝试丢一次」的固定间隔(`period = max(2, round(1.0 / drop_rate))`),无法表示 75% 这类比率。修复改为在决策之间携带小数部分(分数丢弃),使累计丢弃数与计算目标相差不超过 1 次 store。

## 修复方案

修改 `vllm/v1/kv_offload/tiering/backpressure.py` 的 `ThrottledDropPolicy`:用浮点 `_drop_accumulator` 替代 `_call_count`;`should_store` 中把 `drop_rate` 累加进累加器,达到 1.0 即丢弃并减去 1.0,余数带入下次决策;`reset` 清零累加器。新增 40%/75%/变率回归测试与 manager 级测试(8 次 store、75% 率 → 2 次提交、6 次丢弃、12 个 dropped blocks、pin 释放)。

```diff
-        self._call_count += 1
-        period = max(2, round(1.0 / drop_rate))
-        return (self._call_count % period) != 0
+        self._drop_accumulator += drop_rate
+        if self._drop_accumulator >= 1.0:
+            self._drop_accumulator -= 1.0
+            return False
+        return True
```

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:仅 `ThrottledDropPolicy` 的丢弃判定序列与 reset 语义(旧测试 `test_throttled_reset_also_clears_call_count` 被替换为验证 reset 后同序列可复现的 `test_throttled_reset_restarts_decisions`);EMA 压力检测、冷却、pin 释放等行为不变;不影响推理请求准入。正文声明不验证吞吐/延迟收益。

## 关联

#50045(related)、#51576、#51240

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
