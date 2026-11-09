# PR #20999: Replace clamp_position with JIT kernel + platform dispatch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @merrymercy |
| 合并 | 2026-03-22 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | documentation, run-ci, jit-kernel |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 含新 JIT kernel(.cuh + Python wrapper)、platform dispatch、64-case 测试、benchmark 及一个 profiling skill 文档 |
| 链接 | https://github.com/sgl-project/sglang/pull/20999 |

## 一句话结论

把 decode 位置计算 `clamp(seq_lens-1, min=0)` 从 torch.compile 换成专用 CUDA JIT kernel,H100 上大 batch 时最快 4.18x(32768 元素:6.94us→1.66us)。

## 动机

`clamp_position` 是每个 decode step 都要算的位置张量(clamp(seq_lens - 1, min=0)),此前用单一 `@torch.compile(dynamic=True)` 实现:小 batch 下有 compile wrapper 的固定开销,大 batch(16K+)下 torch.compile 生成的实现膨胀明显(16384 时 4.27us,32768 时 6.94us),而 plain PyTorch 只要约 3.5us——这本来就该是一个无分支的 elementwise kernel。

## 优化点

新增 `clamp_position.cuh`:按 dtype(int32/int64)模板化的极简 CUDA kernel(减一后负数取零,256 线程/块,经 sgl_kernel 的 TensorMatcher/LaunchKernel 基础设施 launch),Python 侧用 `@cache_once` 按 dtype 缓存 JIT module。`forward_batch_info.py` 中的单一 torch.compile 实现改为平台 dispatch:CUDA 用 JIT kernel,HIP/AMD 继续用 torch.compile(dynamic),其他平台(NPU、CPU 等)用 plain torch 原生实现。附带提交一个 `.claude/skills/generate-profile` 的 e2e profiling skill 文档(与性能无直接关系)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/clamp_position.cuh: 模板化 CUDA kernel(int32/int64)
python/sglang/jit_kernel/clamp_position.py: @cache_once JIT wrapper,tvm_ffi module
python/sglang/srt/model_executor/forward_batch_info.py: clamp_position 改平台 dispatch(CUDA JIT / HIP compile / 其他 plain torch)
python/sglang/jit_kernel/tests/test_clamp_position.py: 64 个参数化用例
python/sglang/jit_kernel/benchmark/bench_clamp_position.py: triton.testing 三方对比 benchmark
```

关键 dispatch 片段:

```python
if is_cuda():
    clamp_position = clamp_position_cuda          # CUDA JIT kernel
elif is_hip():
    clamp_position = torch.compile(_clamp_position_native, dynamic=True, ...)
else:
    clamp_position = _clamp_position_native       # plain torch
```

## 性能声明

PR 注明条件:NVIDIA H100,seq_lens 为 int64 随机张量,JIT vs torch.compile vs PyTorch 三方对比(单位 us):

| size | JIT | torch.compile | PyTorch | JIT vs compile |
|---|---|---|---|---|
| 16 | 1.35 | 1.56 | 3.36 | 1.15x |
| 128 | 1.55 | 1.61 | 3.40 | 1.04x |
| 1024 | 1.59 | 1.71 | 3.42 | 1.08x |
| 4096 | 1.63 | 2.20 | 3.46 | 1.35x |
| 16384 | 1.66 | 4.27 | 3.52 | 2.57x |
| 32768 | 1.66 | 6.94 | 3.57 | 4.18x |

JIT kernel 时间随 size 几乎平坦(1.35→1.66us);PR 未提供 E2E server 级数字,收益定位在这条每步热路径的 kernel 开销。

## 验证与回归风险

单测 64/64 通过:8 个 size(含 127/255/4097 等非对齐边界)× 2 dtype × 4 场景(normal/zeros/ones/mixed),全部与 `torch.clamp(seq_lens-1, min=0)` reference 严格相等;e2e `test/registered/core/test_srt_engine.py` 8/8 通过。风险:CUDA 路径行为从 torch.compile 的动态 shape 换成固定 dtype 模板,仅支持 int32/int64(其他 dtype 走 `torch.empty_like` 后会由 TensorMatcher 拦截);HIP 平台行为不变,NPU/CPU 从 `disable=_is_npu` 的 compile 变为 plain torch(语义相同);首次调用有 JIT 编译开销(模块缓存)。

## 关联

同区域:sglang.jit_kernel 基础设施(cache_once/load_jit/make_cpp_args)的其他 kernel;`compute_position_torch` 等相邻位置计算路径未改动。同分类其他 PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
