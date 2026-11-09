# PR #21917: [Core] Optimize scheduler request removal for single completions

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chi2liu |
| 合并 | 2025-08-20 |
| 分类 | Overlap / Scheduler |
| Label | ready, v1 |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/21917 |

## 一句话结论

v1 scheduler 请求移除加单请求 fast path:单请求移除 2.6–3.4×,scheduler 性能 +18.7%。

## 动机

v1 scheduler 的 `update_from_output` 里,只要本轮有请求停止,就对整个 running 列表做一次列表推导重建(`[req for req in self.running if req not in stopped_running_reqs]`);`finish_requests` 则对每个请求逐个 `list.remove()`。而最常见的情形恰好是单请求完成,这时整表重建/逐个删除都是不必要的 O(n) 开销;按 PR 的生产口径,200 req/s 下每小时累计浪费约 560ms。

## 优化点

在 `vllm/v1/core/sched/utils.py` 新增 `remove_all(lst, items_to_remove)`,按待移除集合大小分流:恰一项(最常见)时直接 `lst.remove(item)` 原地删除并返回原列表,省去整表扫描重建;多项时回退到原来的列表推导(`item not in set`)。`update_from_output` 的 stopped_running_reqs 移除与 `finish_requests` 的 running 移除均改走该函数,后者同时把 `running_requests_to_remove` 从 list 改为 set;多项路径与 preempted 等其他路径的逻辑保持不变。

## 关键改动

```text
vllm/v1/core/sched/utils.py: 新增 remove_all(单请求 fast path + 多项列表推导回退)
vllm/v1/core/sched/scheduler.py: update_from_output 与 finish_requests 改用 remove_all
```

```python
if len(items_to_remove) == 1:
    # Fast path for single item removal (most common case)
    item = next(iter(items_to_remove))
    with contextlib.suppress(ValueError):
        lst.remove(item)
    return lst
# For multiple items, use list comprehension
return [item for item in lst if item not in items_to_remove]
```

## 性能声明

- 单请求移除 2.6–3.4× speedup;request removal 操作整体 64% improvement;scheduler 性能 +18.7%。
- 生产影响:200 req/s 下每小时节省 560ms。
- PR 未注明测试条件(未给出模型/硬件/workload 与测量方法)。

## 验证与回归风险

diff 未包含任何测试,PR 正文也无 test plan/test result(checklist 未勾选),素材中未见 CI 佐证。风险面:`remove_all` 单项路径原地修改并返回原列表、多项路径返回新列表,调用方必须使用返回值(diff 中两处调用均已如此);单项 fast path 用 `list.remove` 的首匹配语义,多项路径与原实现同为集合 `not in` 判断,多项场景语义不变;`finish_requests` 的 set 化仅影响去重,不改变移除结果。

## 关联

无(素材未提及前序/后续或同区域 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
