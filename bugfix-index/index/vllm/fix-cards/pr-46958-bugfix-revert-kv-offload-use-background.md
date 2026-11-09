# PR #46958: [BugFix] Revert "[KV Offload] Use background thread for mmap / cpu_tensors pinning"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46958](https://github.com/vllm-project/vllm/pull/46958) |
| 作者 | varun-sundar-rabindranath |
| 合并日期 | 2026-06-29 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | KV Cache / Memory |
| bug 类型 | regression |

## 一句话结论
回退「KV Offload 后台线程 pinning」改动,恢复 CPU offload 内存在分配/初始化阶段同步 pin(mmap 用 cudaHostRegister),解决 issue #46933。

## 问题现象
正文未描述具体现象,仅声明 "Fixes https://github.com/vllm-project/vllm/issues/46933"。Test Plan 与 Test Result 小节为空,无验证信息。

## 根因
正文未披露根因分析,只给出对应 issue 链接。由 diff 推断:被回退的方案把 CPU offload 内存的 pin 操作放到后台线程——CUDA/ROCm 上 `torch.zeros` 先以 `pin_memory=False` 分配、由 `CPUTensorPinThread` 异步 `cudaHostRegister`,handler 需携带 `pin_thread` 与 `manually_pinned_tensors` 并在 shutdown 时 join 线程、逐一 `cudaHostUnregister`;回退恢复为分配时直接 `pin_memory=True`、mmap 区域在初始化时同步 pin,删除了整条后台线程生命周期路径。具体故障表现以 issue #46933 为准(本 pack 未含 issue 正文)。

## 修复方案
按 revert 撤销后台 pinning 改动,`vllm/v1/kv_offload/cpu/gpu_worker.py`:

1. 新增 `pin_mmap_region()`:mmap 存在且开启 pin_memory 时,初始化阶段同步调用 `cudaHostRegister` 注册整个 mmap 区域,失败则 WARNING 提示传输仍可用但为 unpinned DMA:
```diff
+def pin_mmap_region(region: SharedOffloadRegion) -> None:
+    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
...
+    base_ptr = region._base.data_ptr()
+    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
```
2. CPU tensor 分配恢复 `pin_memory=pin_memory`(删除「CUDA/ROCm 异步注册故先不 pin」的特判)。
3. 删除 `_pin_cpu_tensors` 后台线程、`SingleDirectionOffloadingHandler` 的 `pin_thread`/`manually_pinned_tensors` 参数,以及 shutdown 中对应的 join 与 `cudaHostUnregister` 清理逻辑。

## 影响与回归风险
- 影响窗口:自引入后台 pinning 的改动起至本回退合并前的版本受影响。
- 回归面:仅影响 `vllm/v1/kv_offload/cpu/gpu_worker.py` 的 CPU KV offload 初始化与关闭路径;worker 初始化期间同步 pin 大块内存(原文被回退方案注释称「Pinning here would block worker initialization」),启动耗时可能回升。正文未提供测试结果,回归风险以 issue #46933 讨论为准。

## 关联
- [issue #46933](https://github.com/vllm-project/vllm/issues/46933)(正文 "Fixes" 引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
