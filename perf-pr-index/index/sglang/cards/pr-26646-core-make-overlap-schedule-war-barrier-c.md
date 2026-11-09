# PR #26646: [core] Make overlap-schedule WAR barrier CUDA-only

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @bingxche |
| 合并 | 2026-05-29 |
| 分类 | Overlap / Scheduler(major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/26646 |

## 一句话结论

把 overlap 调度的 WAR barrier 限定在 CUDA(#26380 保留),HIP/NPU/CPU 回退无 barrier 行为,恢复 AMD 上约 20% 的吞吐损失并修复挂死。

## 动机

#26380 为修复 overlap 调度的数据竞争,在每轮调度前加了全局 WAR barrier(`schedule_stream.wait_stream(forward_stream)`)。该 barrier 在 CUDA 上无吞吐损失,但在 AMD/HIP 上引入三个问题:`test_deepseek_r1_mxfp4_8gpu`(MTP)全部 8 个 TP rank 触发 scheduler watchdog 超时(挂死);`test_priority_scheduling_request_ordering_validation` 断言失败;DSv4 非 MTP 吞吐下降约 20%。本 PR 起于对 #26380 的 revert,最终方案是按平台拆分:CUDA 保留数据竞争修复,其他平台回退到 #26380 之前的行为。

## 优化点

用 `self._war_barrier_enabled = is_cuda()` 作为开关,给三处 `schedule_stream.wait_stream(forward_stream)` barrier(scheduler 主事件循环、disaggregation decode、disaggregation prefill)加门控;CUDA 以外的 HIP/NPU/CPU 不再等待,恢复无 barrier 的重叠调度。同时把 `overlap_utils.py` 中配套的 `fwd_prepare_d2h_stream` + pinned host buffer 补偿机制从 `_is_cuda or _is_hip` 收窄到 `_is_cuda`——该补偿是为找回 WAR barrier 占用的 D2H stream 占用而设的,无 barrier 的平台走普通 `.cpu()` 路径即可。CUDA 侧行为完全不变。

## 关键改动

```text
python/sglang/srt/managers/scheduler.py: _war_barrier_enabled = is_cuda();主循环 barrier 加门控
python/sglang/srt/disaggregation/decode.py: disagg decode 事件循环 barrier 加门控
python/sglang/srt/disaggregation/prefill.py: disagg prefill 事件循环 barrier 加门控
python/sglang/srt/managers/overlap_utils.py: pinned-buffer/D2H-stream 补偿收窄到 CUDA-only
```

```python
# scheduler.py
# WAR barrier is CUDA-only; other platforms keep the pre-barrier behavior.
self._war_barrier_enabled = is_cuda()
...
# WAR barrier: this iter's schedule writes to shared GPU buffers
# wait for prev forward's reads.
if self._war_barrier_enabled:
    self.schedule_stream.wait_stream(self.forward_stream)
```

## 性能声明

PR 正文引用 #26380 的数据(https://github.com/sgl-project/sglang/pull/26380#issuecomment-4571064550):

- CUDA 上 barrier 无可测吞吐变化(Llama-3 + EAGLE3 MTP、DSv4 MTP、DSv4 非 MTP);本 PR 不改变 CUDA 行为。
- AMD/HIP 上 #26380 造成 DSv4 非 MTP 约 20% 吞吐下降,本 PR 在非 CUDA 平台移除 barrier 后恢复;`test_deepseek_r1_mxfp4_8gpu` 的 8-rank 挂死与 priority scheduling 断言失败随之修复。

未提供本 PR 自身的逐项 benchmark 表;测试条件(DSv4/HIP 硬件型号等)未在正文中注明。

## 验证与回归风险

PR 正文未列新增单元测试(Checklist 未勾选),依赖其引用的既有测试:`test_deepseek_r1_mxfp4_8gpu`(MTP,8 GPU)、`test_priority_scheduling`、以及 CUDA 侧 Llama-3 + EAGLE3(MTP)/DSv4 MTP/非 MTP 的吞吐对照。回归风险的核心是数据竞争窗口:非 CUDA 平台回到 barrier 之前的调度行为,#26380 修的 overlap 调度 WAR 冲突(req_to_token_pool / SWA mapping 等共享 GPU buffer 的 schedule 写 vs forward 读)在 HIP/NPU/CPU 上重新暴露,正确性依赖这些平台原有的隐式同步(如 `.cpu()` 路径);CUDA-only 的 pinned-buffer 补偿收窄不影响其他平台,但需确认 HIP 无残余引用。

## 关联

直接关联 #26380(WAR barrier 的引入,本 PR 部分保留、部分回退)。pack 未列出其他同分类 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
