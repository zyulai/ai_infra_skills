# PR #19880: [JIT Kernel][Feature] Support JIT custom all reduce (rewrite as v2)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-03-20 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 可用,另有 11 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/19880 |

## 一句话结论

JIT 化的 custom all reduce v2:可配置 SM 数/CTA size,融合 PDL 与 push-mode 1-shot,小 batch 下延迟最高降 40%。

## 动机

PR 正文的 Motivation 段为空(仅保留模板注释)。从 Modifications 可知目标:重写一版干净、高度可配置的 custom all reduce(可设置 SM 数量、推荐 CTA size),并把 post-Hopper 特性(如 PDL,Programmatic Dependent Launch)整合进来,改善小 batch 下的 all-reduce 延迟;另引入 push-mode 1-shot all reduce,在小 batch 下显著快于 pull-mode。

## 优化点

- 以 JIT 方式(`load_jit` + tvm_ffi)编译 three 种算法:`ONE_SHOT_PULL`、`TWO_SHOT_PULL`、`ONE_SHOT_PUSH`,按 dtype/world_size/PDL 支持与否实例化 CUDA 模板,替代 AOT 预编译的 v1 kernel。
- pull 路径从 AOT kernel(sgl-kernel v0.5.9,源自 vllm v0.8.2)迁移并重新设计 controller 接口以减少控制面流量,2-shot 中把 reduce-scatter 与 broadcast 融合进一个 kernel;`config_pull` 允许运行时调整 grid/block 维度上限(仅 pull 可配)。
- push 路径改写自 flashinfer 的 trtllm_allreduce_fusion,简化 Lamport 设计并把 ring buffer 从 3 个减到 2 个:Phase 1 把输入 push 到各 rank buffer(正零清成负零作标志位),Phase 2 用 volatile 16B load 轮询并归约,写回输出后重置标志。
- kernel 全部支持 PDL(`PDLWaitPrimary`/`PDLTriggerSecondary`),CUDA graph 捕获时 1-shot pull 通过 `allocate_graph_capture_input` 分配额外输出 buffer,push 模式 block 数必须为常量以保证正确性。

## 关键改动

```text
python/sglang/jit_kernel/all_reduce.py: AllReduceAlgo 枚举、CustomAllReduceObj FFI 包装与 JIT 模块加载
python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_pull.cuh: one/two-shot pull kernel(迁移自 AOT,重设计 controller)
python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh: push-mode 1-shot kernel(改写自 flashinfer,Lamport 简化、ring buffer 3→2)
python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_base.cuh: FFI 对象注册
python/sglang/jit_kernel/benchmark/bench_custom_all_reduce.py: JIT vs NCCL vs AOT v1 vs FlashInfer 对比基准
```

## 性能声明

PR 正文仅给出两点:整合 PDL 后 custom-all-reduce 延迟在小 batch size 下提升最高 40%;push-mode 1-shot 在小 batch 下显著快于 pull-mode。正文的 Accuracy Tests 与 Benchmarking 段落为空,PR 未注明测试条件(硬件、world size、消息大小、具体数字表);diff 中的 `bench_custom_all_reduce.py` 提供 4K–32M 消息、bfloat16/float16/float32、CUDA graph 计时回放的基准脚本,但 pack 中未附运行结果。

## 验证与回归风险

PR 正文未提供 accuracy 测试结果(该段落为空)。仓库内新增的基准脚本用 torchrun 覆盖 2/4/8 GPU,聚合各 rank 最大耗时后与 NCCL、AOT v1、FlashInfer fusion 对比,可用于回归观察。风险点:依赖 JIT 编译基础设施(tvm_ffi、`-lcuda`);PDL 仅在支持的架构上启用(`is_arch_support_pdl`);push 模式只支持 1-shot、要求输入 contiguous 且 16B 对齐、元素数上限 4G;push buffer 大小限制输入消息规模;2-shot pull 要求消息长度可整除向量宽度。

## 关联

前序:AOT 版 custom all reduce v1(`sgl-kernel/csrc/allreduce/custom_all_reduce.cu`,源自 vllm v0.8.2);同类对比对象:NCCL 与 flashinfer trtllm allreduce fusion。pack 未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
