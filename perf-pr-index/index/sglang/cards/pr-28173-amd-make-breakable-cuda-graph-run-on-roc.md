# PR #28173: [AMD] Make breakable CUDA graph run on ROCm/HIP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @oulgen |
| 合并 | 2026-06-19 |
| 分类 | CUDA Graph / Compiler |
| Label | documentation, run-ci, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/28173 |

## 一句话结论

把 breakable CUDA graph(分段捕获)从 NVIDIA 扩展到 ROCm/HIP,AMD GPU 也能用分段图捕获。

## 动机

Breakable CUDA graph(BCG,分段捕获、无 torch.compile)此前在实现里直接 `raise RuntimeError("Breakable CUDA graph is not supported on ROCm/HIP")`,`--debug-cuda-graph` 在非 CUDA 平台一律被禁用。BCG 依赖 cuda-python 查询流的 capture status,HIP 平台没有 cuda-python,需要一条可移植的等价查询路径才能在 AMD 上启用该能力。

## 优化点

核心是把流捕获状态查询抽象为 `_is_stream_capturing(stream)`:ROCm/HIP 走可移植的 `torch.cuda.is_current_stream_capturing()`(映射到 HIP runtime),NVIDIA 保持原样用 cuda-python 直查 CUDA runtime——因为 torch 的可移植 API 在 CUDA 上被证实不可靠。由此移除 backend `__init__` 的 HIP 拒绝、`_hooked_wait_stream` 与 `_end_current_segment` 中所有直接调 `_capture_status` 的位置改走新函数;server_args 的 `--debug-cuda-graph` 放行条件从 is_cuda() 扩为 is_cuda() or is_hip();`_disable_breakable_cudagraph_if_incompatible` 的注释去掉 HIP 拒绝。配套改动:triton 与 aiter attention 后端的 forward_extend 在 `forward_batch._attn_output` 存在时复用该输出 buffer(BCG 提供的缓冲),避免每层分配新输出(aiter 路径额外要求 q 非 fp8);文档 Compatibility 一节改为 "CUDA and ROCm/HIP",NPU/CPU/MPS/XPU 仍不支持并自动禁用加警告。

## 关键改动

```text
python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py: _is_stream_capturing 按平台分流(HIP 用 torch API)
python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py: 移除 HIP RuntimeError
python/sglang/srt/server_args.py: --debug-cuda-graph 放行 HIP;非 CUDA/HIP 仍禁用
python/sglang/srt/layers/attention/triton_backend.py / aiter_backend.py: 复用 forward_batch._attn_output 输出缓冲
test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py: 注册 AMD CI,移除 cuda-python 跳过守卫
```

## 性能声明

PR 正文为空,无性能数字。文档(diff 上下文可见)称 BCG 对"少量 graph break 的典型场景"开销可忽略;本 PR 的价值是把该能力带到 AMD 平台,素材未提供 AMD 上的对比测量。

## 验证与回归风险

既有注册测试 test_breakable_cuda_graph.py 新增 `register_amd_ci`(stage-c-test-large-8-gpu-amd-mi35x,est_time=120)并移除 cuda-python 的 skip 守卫,即在 AMD MI35X 8 卡 CI 上直接跑 BCG 集成测试;CUDA CI 维持原注册。回归面:NVIDIA 路径的 capture-status 查询行为刻意保持不变;HIP 路径改用 torch 可移植 API(在 stream 上下文中调用),若 torch 该 API 在 HIP 上行为有偏差会影响 forked-stream 的段收尾逻辑;triton/aiter extend 的输出 buffer 复用仅在 BCG 注入 `_attn_output` 时生效,常规路径不受影响;memory saver 模式仍不兼容。

## 关联

前序:breakable CUDA graph 的 NVIDIA 原始实现(编号素材未给出);同区域为 CUDA Graph / Compiler 类的 piecewise/breakable 图捕获工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
