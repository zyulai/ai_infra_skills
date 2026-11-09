# PR #29207: use the same stream for cuda graph catpure and replay for NCCL

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Amir-19 |
| 合并 | 2025-12-25 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | performance, ready, v1, nvidia |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 4 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/29207 |

## 一句话结论

让 CUDA Graph capture 与 warmup/replay 复用同一 per-process stream,避免 NCCL 2.28+ 在 capture 期间崩溃并减少多余内存注册。

## 动机

NCCL 2.28+ 引入 window registration 后,`ncclMemAlloc` 分配的内存与 stream 绑定。vLLM 此前 warmup 迭代与 CUDA Graph capture 使用不同 stream:capture 时若 associated mempool 中没有可用 segment,就需要新的内存分配,进而触发新的 NCCL window registration,导致 capture 期间 NCCL 直接报错崩溃。因此需要保证 capture 使用的 stream 与 warmup 一致,复用已注册的内存窗口。

## 优化点

核心改动是把"专用 stream"策略从 ROCm 扩展到 CUDA:`vllm.utils.torch_utils.current_stream()` 在首次调用时为整个进程创建一个 dedicated stream(CUDA 上也不能用 default stream 做 graph capture,PyTorch 的 CUDAGraph 捕获不支持默认 stream)。随后在 `vllm/compilation/cuda_graph.py` 里,`torch.cuda.graph(...)` 显式传入 `stream=current_stream()`,强制 capture 发生在与 warmup 迭代相同的 stream 上。这样 capture 阶段命中已有 mempool segment,不再触发新的 `ncclMemAlloc` + `ncclCommWindowRegister`。benchmark 脚本中的 capture 路径同步修改以保持一致。

## 关键改动

```text
vllm/compilation/cuda_graph.py: CUDA Graph capture 时显式传入 stream=current_stream()
vllm/utils/torch_utils.py: current_stream() 在 CUDA(原仅 ROCm)上也创建 per-process 专用 stream
tests/utils_/test_torch_utils.py: 测试统一为 CUDA/ROCm 均断言专用 stream 语义
benchmarks/kernels/benchmark_device_communicators.py: benchmark 的 capture 传入同一 stream
```

关键片段(vllm/compilation/cuda_graph.py):

```python
with torch.cuda.graph(
    cudagraph,
    pool=self.graph_pool,
    stream=current_stream(),
):
    output = self.runnable(*args, **kwargs)
```

## 性能声明

PR 正文未给出 benchmark 数字(perf claims 检测为 none);性能收益体现为消除 NCCL 2.28+ 下 capture 时的崩溃与多余的内存分配/注册开销。PR 未注明测试条件。

## 验证与回归风险

依赖 `tests/utils_/test_torch_utils.py::test_current_stream_multithread`,该测试从"ROCm 专用 stream / CUDA 默认 stream"分支改为统一断言 CUDA 与 ROCm 都返回同一专用 stream。风险面:所有使用 `current_stream()` 的 CUDA 代码路径从 default stream 切到 per-process dedicated stream,可能影响依赖 default stream 语义的外部交互(如 NCCL 通信、第三方库隐式同步);CUDA Graph capture/replay 行为随 stream 改变,需覆盖 NCCL window registration 场景。

## 关联

同属 CUDA Graph / Compiler 分类;与 `benchmarks/kernels/benchmark_device_communicators.py` 覆盖的 device communicator 优化相关。PR 未注明前序/后续依赖。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
