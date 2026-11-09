# PR #7685: fix: prevent crashes due to logit bias dimension mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#7685](https://github.com/sgl-project/sglang/pull/7685) |
| 作者 | 0xymoro |
| 合并日期 | 2025-07-23 |
| labels | high priority |
| 热度 | 👍 4 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
batch 合并时 logit_bias 在 temperatures 更新后才 merge 导致维度错位崩溃;EAGLE verify 补上 sampling_info 过滤,两处一起修。

## 问题现象
复现条件(正文照抄):having high batchsize of requests that have frequency/repetition penalty。即大批量、带 frequency/repetition penalty 的请求下发生 race condition 导致崩溃(issue #7585)。作者声明 PR 在与上述相同条件下验证并消除了该 bug,正文未附具体崩溃日志。

## 根因
正文披露:logit bias 没有在正确的位置设置——`merge_batch` 中 `logit_bias` 的合并被放在 temperatures 张量更新之后执行,而按代码对 `len(self)`/`len(other)` 的注释要求,必须先于 temperatures 的 merge 完成,否则维度错位;另外 EAGLE verification 内部还需要保证所用的 batchsize 与 sampling_info 同步(此时二者可能不一致)。

## 修复方案
- `sampling/sampling_batch_info.py::merge_batch`:把 `logit_bias` 的合并移到 temperatures 张量更新之前。
- `speculative/eagle_utils.py::verify`:当 `bs != len(sampling_info)` 时 deepcopy `sampling_info` 并按 `retrive_index` 调 `filter_batch`,保证维度一致。

```python
+        if bs != len(sampling_info):
+            sampling_info = copy.deepcopy(sampling_info)
+            # NOTE: retrive_index are the indices of the requests that are kept.
+            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)
```

## 影响与回归风险
- 影响窗口:正文未说明
- 回归面:`merge_batch` 的合并顺序(所有批合并路径)与 EAGLE verify 的 sampling_info 维护;未启用 logit_bias/EAGLE 的路径不受影响。

## 关联
- https://github.com/sgl-project/sglang/issues/7585

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
