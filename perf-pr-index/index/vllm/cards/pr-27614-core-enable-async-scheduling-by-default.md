# PR #27614: [Core] Enable async scheduling by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2025-12-29 |
| 分类 | Overlap / Scheduler(major) |
| Label | structured-output, frontend, ready, v1, gpt-oss, kv-connector, ready-run-all-tests |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/27614 |

## 一句话结论

把 V1 异步调度(async scheduling)从默认关闭改为默认开启,减少 GPU 利用率空隙,不兼容场景自动回退同步调度。

## 动机

异步调度可避免 CPU 调度与 GPU 执行串行化造成的 GPU 利用率空隙,改善延迟与吞吐,但此前默认关闭(`async_scheduling: bool = False`),需要用户显式开启。该 PR 是在所有阻塞问题(关联 issue #27679)解决后将其转为默认行为。

## 优化点

`async_scheduling` 默认值改为 `None`(三态:未显式配置时自动决定):`pipeline_parallel_size > 1` 时禁用并告警;speculative decoding 时禁用并区分提示——非 EAGLE/MTP 方法提示"不支持",EAGLE/MTP 方法提示"部分特性与投机解码不兼容,可显式 `async_scheduling=True` 强制开启";executor 不支持时禁用。开启时自动 `disable_nccl_for_dp_synchronization=True`(DP 同步不走 NCCL),并用 `info_once` 打印最终 enabled/disabled 状态。显式 `True/False` 配置仍然生效。

## 关键改动

```text
vllm/config/scheduler.py: async_scheduling 默认 None,文档改为"设 False 关闭"
vllm/config/vllm.py: __post_init__ 按兼容性自动决定;DP 下禁用 NCCL 同步;打印状态
```

## 性能声明

PR 正文未给出 benchmark 数字,仅说明阻塞问题已解决并通过扩展 CI 测试。异步调度本身的收益表述来自配置文档:"helps to avoid gaps in GPU utilization, leading to better latency and throughput"。

## 验证与回归风险

测试:通过扩展 CI(run-all-tests label);pack 未列出新增单测。回归风险:所有此前默认同步调度的部署行为发生变化——投机解码(即使 EAGLE/MTP)、PP>1、executor 不支持的场景自动回退,但其余路径默认多了一层异步调度语义;structured-output、kv-connector 等 label 相关路径此前已适配;用户可用 `async_scheduling=False` 恢复旧行为;显式 True 配投机解码时不再走自动禁用分支(需自行承担兼容性)。

## 关联

前置阻塞 issue #27679;同区域为 V1 调度器 overlap 优化;pack 正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
