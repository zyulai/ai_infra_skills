# PR #20976: Replace _resolve_future_token_ids with JIT kernel + platform dispatch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @merrymercy |
| 合并 | 2026-03-20 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 5 个文件:1 个 .cuh kernel、1 个 Python JIT wrapper、1 个测试、1 个 benchmark、overlap_utils.py 调用点改造 |
| 链接 | https://github.com/sgl-project/sglang/pull/20976 |

## 一句话结论

overlap 调度里 `_resolve_future_token_ids` 从 torch.compile 换成专用 CUDA JIT kernel,H100 上较 torch.compile 最高快 1.30x、较纯 PyTorch 快约 4x。

## 动机

在 overlap 调度路径中,`_resolve_future_token_ids` 负责把 input_ids 中的负值(未来 token 占位)按 `future_token_ids_map[-value]` 就地解析为真实 token id。原实现是单个 `@torch.compile` 装饰的 torch 组合操作,对这种简单 elementwise scatter 存在编译产物与执行开销;不同硬件平台对 torch.compile 的支持程度也不一致,需要更快的专用实现与清晰的平台分发。

## 优化点

新增一个模板化 CUDA JIT kernel(`resolve_future_token_ids.cuh`,支持 int32/int64):每个线程检查一个元素,负值就地替换为 `future_map[-val]`(带溢出 clamp),非负值保持不变;host 侧用 TensorMatcher 校验 dtype/device,按 `kBlockSize=256` 切 grid。Python 侧通过 `@cache_once` 缓存 JIT 模块,按 dtype 编译一次后复用。调用点改为平台分发:CUDA 走 JIT kernel,HIP/AMD 继续用 torch.compile,其他平台(NPU、CPU 等)用原生 torch 实现,避免单一 `@torch.compile` 方案的平台兼容问题。

## 关键改动

```text
python/sglang/jit_kernel/csrc/elementwise/resolve_future_token_ids.cuh: CUDA JIT kernel
  本体(模板化 dtype,负值 in-place 查表替换)
python/sglang/jit_kernel/resolve_future_token_ids.py: JIT wrapper,@cache_once 按 dtype
  编译并缓存模块,暴露 resolve_future_token_ids_cuda
python/sglang/srt/managers/overlap_utils.py: 调用点改为平台分发(cuda=jit / hip=torch.compile
  / 其他=原生 torch)
python/sglang/jit_kernel/tests/test_resolve_future_token_ids.py: 单元测试(多种 size、
  dtype、全负/全非负/混合/零值输入)
python/sglang/jit_kernel/benchmark/bench_resolve_future_token_ids.py: JIT vs torch.compile
  vs PyTorch 三方 benchmark
```

关键摘录(kernel 本体):

```cuda
__global__ void resolve_future_token_ids_kernel(
    T* __restrict__ input_ids, const T* __restrict__ future_map, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val = input_ids[idx];
    if (val < 0) {
      T key = -val;
      if (key < 0) key = 0;  // clamp for overflow
      input_ids[idx] = future_map[key];
    }
  }
}
```

## 性能声明

NVIDIA H100 benchmark(map_size=8192,单位 us):

| size | JIT kernel | torch.compile | PyTorch | JIT vs torch.compile |
|---|---|---|---|---|
| 16 | 2.82 | 3.25 | 12.31 | 1.15x |
| 128 | 3.23 | 3.27 | 12.30 | 1.01x |
| 1024 | 3.31 | 3.47 | 14.16 | 1.05x |
| 4096 | 3.37 | 3.51 | 14.79 | 1.04x |
| 16384 | 3.57 | 4.02 | 15.22 | 1.13x |
| 32768 | 3.68 | 4.79 | 15.57 | 1.30x |

PR 未注明端到端 serving 吞吐数字。

## 验证与回归风险

单元测试 64/64 通过(8 种 size x 2 种 dtype x 4 类输入模式:全负/全非负/混合/零,与纯 torch 参考实现逐值比对);e2e 测试 `test/registered/core/test_srt_engine.py` 8/8 通过。回归风险:HIP/AMD 与 NPU/CPU 路径行为不变(仍走 torch.compile/原生 torch);CUDA 路径从编译图切换到手写 kernel,语义覆盖测试已含 clamp 边界,但依赖 `sglang.jit_kernel` 的编译基础设施可用。

## 关联

无(pack 内未提及同区域其他 PR;属于 sgl jit_kernel 体系建设的一部分)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
