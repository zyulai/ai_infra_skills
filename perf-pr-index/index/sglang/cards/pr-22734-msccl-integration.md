# PR #22734: MSCCL++ Integration

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @caiocbr |
| 合并 | 2026-06-09 |
| 分类 | Other (tier: major) |
| Label | documentation, sgl-kernel, run-ci, piecewise-cuda-graph, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用(另有 9 个文件因超长未包含在 pack 中) |
| 链接 | https://github.com/sgl-project/sglang/pull/22734 |

## 一句话结论

将 MSCCL++ allreduce 从 sgl-kernel 静态 C++ 内核迁移到上游 mscclpp Python 包并加自动调优,多节点 TP 推理吞吐最高提升 3.4×。

## 动机

旧 MSCCL++ 集成以静态链接 C++/CUDA kernel 形式打包在 sgl-kernel 内:手写 LL allreduce 内核版本固定、难以扩展新 collective 算法,仅支持 TP=8/16 两档规模;且与 piecewise CUDA graph 编译存在交互问题——在 compile/capture 阶段走 MSCCL++ dispatch 会触发 `torch.compile` guard 失效与重编译。

## 优化点

1. **迁移到上游 mscclpp Python 包**:重写 `PyMscclppCommunicator` 为纯 Python 实现,改用 `mscclpp` 的 Executor、DSL compiler(`mscclpp.language`)与 native algorithm collection,删除 sgl-kernel 中约 900 行手写 CUDA 内核及 FetchContent 依赖。
2. **多算法组合**:TP=8 用四种 native 算法(`nvls_packet`/`packet`/`rsag_zero_copy`/`nvls_zero_copy`)按消息大小分段;TP=16/32(两/四节点)用 DSL 编译的 `allreduce_multi_nodes` 多配置实例。
3. **CUDA graph 自动调优**:初始化时对 512B–8MiB 的 2 的幂消息尺寸,遍历候选算法 × nblocks × nthreads 配置,用 CUDA graph capture 计时并在 rank 间广播同步结果,按消息大小缓存最优配置。
4. **piecewise CUDA graph 兼容**:`should_mscclpp_allreduce()` 检查 `is_in_piecewise_cuda_graph()` 等 PCG 状态,避免编译/捕获期改变 allreduce dispatch 路径。
5. **对称内存(NVLS)支持**:`--enable-symm-mem` 时启用 `default_allreduce_nvls_zero_copy`;`all_reduce()` 改为原地执行。

## 关键改动

```text
python/sglang/srt/distributed/device_communicators/pymscclpp.py: 重写为纯 Python
  communicator;算法创建(_create_native_algorithms/_create_dsl_algorithms)、
  _tune() CUDA graph 调优、destroy();支持规模扩到 TP=8/16/32
python/sglang/srt/distributed/parallel_state.py: all_reduce dispatch 重排,先算
  should_use_pymscclpp_allreduce 再决定 ca/qr/mscclpp 路径;GroupCoordinator.destroy()
  增加 pymscclpp_comm.destroy()
sgl-kernel/csrc/allreduce/mscclpp_allreduce.cu/.cuh + CMakeLists.txt: 删除静态
  C++/CUDA 内核与 mscclpp FetchContent 链接
python/sglang/srt/compilation/piecewise_context_manager.py: 新增 piecewise 编译
  完成状态跟踪
benchmark/kernels/all_reduce/README.md: 新增 MSCCL++ 基准与推理使用文档
```

## 性能声明

以下均来自 PR 正文,未独立核实:

- **Kernel 级 allreduce 延迟**(H100 TP=8,bf16,2KiB–1MiB):MSCCL++ graph 模式较 pynccl graph 快 **1.9–2.9×**(≤256KiB 消息;如 2KiB 5.86µs vs 16.84µs)。
- **单节点 H100 TP=8 e2e**(Qwen3-8B,input 256/out 256,bf16):吞吐最高 **1.12×**(bs=32:7204.70 vs 6424.29 tok/s)。
- **两节点 H200 TP=16 e2e**(Qwen3-8B,input 128/out 128):较 native SGLang 最高 **2.7× 吞吐**(bs=32:8648.05 vs 3202.45 tok/s)。
- **四节点 H200 TP=32 e2e**:较 NCCL 最高 **3.4× 吞吐**(bs=64:13162.17 vs 3827.11 tok/s)。
- **NVLS 对称内存**:`--enable-symm-mem` 下 MSCCL++ 较 NCCL+symm mem 在高 batch(256/512)提供 **3–7%** 额外提升。

正确性:用 `benchmark_mscclpp.py` 将 MSCCL++ 输出与 torch/NCCL allreduce 按消息尺寸逐元素对比(bf16)。

## 验证与回归风险

- 正确性依赖 microbenchmark 逐元素对比,PR 未附模型精度评测。
- PR 页面 CI 状态(pack 记录):Latest PR Test (Base) 失败,Latest PR Test (Extra) 通过。
- `parallel_state.all_reduce()` dispatch 顺序变化影响 ca_comm/qr_comm/pynccl/torch_symm_mem 各后端选择路径,需回归多通信后端组合。
- `should_mscclpp_allreduce()` 依赖 piecewise context manager 新增状态,与 torch.compile 的 guard 交互是主要回归面。
- `custom_all_reduce.close()` 加 ops None guard、bench_one_batch 增加 `destroy_model_parallel()`,涉及进程退出/多节点 rank 计算路径。
- MSCCL++ 仅支持 CUDA;旧代码注释明确其在 AMD 未测试,本次迁移后仍限 CUDA。

## 关联

无(pack 素材未提供同区域其他 PR 编号;label 含 piecewise-cuda-graph,与 piecewise CUDA graph 编译链路相关)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
