# PR #21932: [HiSparse] Optimize the scheduling of decode backup.

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @huangtingwei9988 |
| 合并 | 2026-04-07 |
| 分类 | Overlap / Scheduler(major) |
| Label | hicache, ready-to-merge, run-ci |
| Diff | diffstat 缺失(素材未提供统计);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/21932 |

## 一句话结论

HiSparse 把 decode token 的 host 备份从调度关键路径挪到独立 stream,与 forward 重叠,TPOT -5%。

## 动机

overlap 调度下,`prepare_for_decode` 中的 decode token 备份(把上一 token 的 KV 从 device 拷到 host pool)会等待 forward stream 完成,形成显著 CPU bubble:调度器在发出下一轮 forward 前被备份拷贝阻塞。备份本质上是可延迟的后台工作,不应占住 decode 的关键路径。

## 优化点

重构备份时序:备份 work 立即入队到专用的 `decode_backup_stream`,但通过 event 挂在 `decode_forward_done_event` 之后(forward 结束才开始真正拷贝),调度线程不再同步等待;`_eager_backup_previous_token` 中把原同步的 `backup_from_device_all_layer` 移入 backup stream,并在其上 record `_backup_done_event`,对涉及的所有 device tensor 调 `record_stream` 防止提前释放。消费端在两处等待备份完成:`model_runner._forward_raw` 每次 decode forward 前调用 `wait_for_pending_backup()`(等上一轮备份完成再跑本轮 forward),以及 coordinator `request_finished` 释放内存前。这样备份与"下一轮调度/forward 准备"重叠,每轮 decode 只需等上一轮备份,而非本轮。

## 关键改动

```text
python/sglang/srt/managers/hisparse_coordinator.py: 新增 decode_backup_stream + pending backup event 机制
python/sglang/srt/model_executor/model_runner.py: decode forward 前调用 wait_for_pending_backup()
```

```python
with device_module.stream(self.decode_backup_stream):
    self.decode_backup_stream.wait_stream(schedule_stream)
    if self.decode_producer_stream is not None:
        self.decode_backup_stream.wait_stream(self.decode_producer_stream)
    self.mem_pool_host.backup_from_device_all_layer(...)
    self._backup_done_event.record()
...
def wait_for_pending_backup(self) -> None:
    if not self._has_pending_backup:
        return
    self._backup_done_event.wait(device_module.current_stream())
    self._has_pending_backup = False
```

## 性能声明

h20-96g,DeepSeek-V3.2-Exp,TP8,`--enable-hisparse`(top_k 2048,device_buffer 4096),ShareGPT random 8192 输入/1500 输出,request rate 5.6,concurrency 5,30 请求:

| 指标 | before | after |
| --- | ---: | ---: |
| Mean TPOT (ms) | 33.89 | 32.07 |
| Mean ITL (ms) | 33.94 | 32.12 |
| P95 ITL (ms) | 46.76 | 38.50 |
| Output throughput (tok/s) | 129.29 | 135.67 |
| Total token throughput (tok/s) | 835.39 | 876.61 |
| Benchmark duration (s) | 348.05 | 331.69 |

作者结论:E2E TPOT 提升约 5%。TTFT 基本持平(7160.04 → 7168.33 ms)。

## 验证与回归风险

gsm8k(200 题)前后均为 Accuracy 0.975、Invalid 0.000。回归风险:备份被推迟到 forward 结束后,依赖 event 链(decode_forward_done_event → backup → backup_done_event → 下一轮 wait)的正确性;若 request 在备份完成前退出,`request_finished` 现在会先等 pending backup 再释放内存,防止 use-after-free,但所有释放路径都必须记得等待,漏一处即产生悬垂引用;`record_stream` 只对 is_cuda 的张量调用,host tensor 依赖 wait_stream 语义;改动仅影响启用 HiSparse 的部署,不影响常规路径。

## 关联

无(正文未提及其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
