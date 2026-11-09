# RFC #30679: [RFC]: Replace `torch.cuda` API with `torch.accelerator` for better hardware compatiblity.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#30679](https://github.com/vllm-project/vllm/issues/30679) |
| 作者 | jikunshang |
| 创建日期 | 2025-12-15 |
| 状态 | 🟢 open |
| labels | RFC, unstale |
| 热度 | 👍 13 · 💬 26 |
| 分类 | Hardware / Backend |

## 一句话结论
用 PyTorch 新的 torch.accelerator 统一 API 逐步替换 vLLM 中硬编码的 torch.cuda 调用,改善非 CUDA 设备的硬件兼容性。

## 提案动机
vLLM 是支持多硬件后端的框架,但代码中仍有一些 torch.cuda 硬编码调用,这对非 CUDA 兼容设备不友好。PyTorch 新增了 `torch.accelerator` API,可基于平台自动 dispatch。同时应增加 lint 工具避免新代码再引入 torch.cuda 调用(首个示例为 PR #30681)。

## 提出的方案
- 目标时间:迁移在 Q2 结束前(或更早)完成
- 按 torch-2.9.0 的 torch accelerator API 状态列出完整映射表,逐项替换并跟踪 vLLM 侧状态,例如:
  - `torch.cuda.Event` → `torch.Event`(torch 2.9 ready;vLLM 侧 #26985 后由 #47140 跟进)
  - `torch.cuda.device_count/current_device/set_device()` → `torch.accelerator.*`(Done #36145)
  - `torch.cuda.synchronize` → `torch.accelerator.synchronize`(Done #36085)
  - `torch.cuda.empty_cache` → Done(#30681);max_memory_allocated/memory_allocated/memory_stats 等内存统计系列 → Done #37031
  - `torch.cuda.Stream` → `torch.Stream`(torch 侧推迟到 2.11);`with torch.cuda.stream()` → torch 2.10 ready,vLLM 侧 WIP(#33225)
- 部分 API 无统一替代(NA):CUDAGraph、graph_pool_handle、nvtx、profiler、OutOfMemoryError、CUDAPluggableAllocator、MemPool 系列等,保留原状
- 用 lint 工具防止新增代码引入 torch.cuda 调用

## 优化点与收益
正文未给出量化数字。定性收益:非 CUDA 兼容设备(多硬件后端)兼容性提升,平台 dispatch 由 PyTorch 统一 API 承担,减少逐后端分支。

## 现状与落地
仍在讨论/推进中(open)。正文映射表显示多项已完成(device_count、synchronize、empty_cache、内存统计系列、set_device 等),`torch.Stream`/`with torch.cuda.stream()` 等依赖 torch 2.10/2.11 的部分仍 WIP 或待做;整体迁移目标为 Q2 结束前完成。

## 关联
- torch.accelerator API 文档:https://docs.pytorch.org/docs/stable/accelerator.html
- PR #30681
- PR #36145
- PR #37031
- PR #33225

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
