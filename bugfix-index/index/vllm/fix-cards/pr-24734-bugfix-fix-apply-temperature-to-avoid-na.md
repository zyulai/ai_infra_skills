# PR #24734: [Bugfix] fix apply_temperature to avoid nan in probs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24734](https://github.com/vllm-project/vllm/pull/24734) |
| 作者 | courage17340 |
| 合并日期 | 2025-09-25 |
| labels | ready, v1 |
| 热度 | 👍 3 · 💬 10 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
greedy 请求温度存为 -1.0 被直接除产生 NaN probs 并致 flashinfer 采样非法内存访问;贪婪温度视为 1.0 修复。

## 问题现象
正文说明:本 PR 修复 #22180,可解决使用 flashinfer sampling 时多类 illegal memory access(非法内存访问)。Test Plan 与 Test Result 两节为空,正文未描述具体复现步骤、日志或影响面。

## 根因
正文未系统披露机制,只给出修法结论:"Just treat greedy temperature (0.0 now instead of -1.0) as 1.0 when performing apply_temperature is enough"。由 diff 推断:greedy 请求的温度在 `gpu_input_batch.add_request` 中被置为 -1.0(旧注释称避免后续除零),而 `apply_temperature` 对混合批次统一执行 `logits.div_(temp.unsqueeze(dim=1))`,贪婪请求的 logits 被除以 -1;PR 标题所称的 NaN probs 即来自该路径,进而引发 flashinfer sampling 的非法内存访问。

## 修复方案
- `vllm/v1/sample/sampler.py`:`apply_temperature` 新增 `all_random` 参数,非全随机批次先把低于 `_SAMPLING_EPS` 的温度替换为 1.0 再除,避免贪婪请求除零:

```python
if not all_random:
    temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
return logits.div_(temp.unsqueeze(dim=1))
```

- `vllm/v1/worker/gpu_input_batch.py`:`add_request` 中 greedy 请求的 `temperature_cpu[req_index]` 由 -1.0 改为 0.0(注释同步改为 "Should avoid division by zero later when apply_temperature")。

## 影响与回归风险
- 影响窗口:正文未提及具体受影响版本范围。
- 回归面:V1 sampler 的温度施加路径与 GPU input batch 的贪婪温度编码;全随机批次(`all_random=True`)走原路径不变;混合批次的贪婪行 logits 不再被取反。正文未附测试结果。

## 关联
- Fixes [#22180](https://github.com/vllm-project/vllm/issues/22180)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
