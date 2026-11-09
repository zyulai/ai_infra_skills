# PR #32792: [CPU Backend] [Perf] Accelerate tensor-parallel inference across NUMA domains on Arm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fadara01 |
| 合并 | 2026-01-22 |
| 分类 | Parallelism / Communication（tier: major） |
| Label | ready, ci/build, v1, cpu |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/32792 |

## 一句话结论

Arm 上启用共享内存通信器与自动绑核，tp=2/双 NUMA 吞吐较现状约 4x（c8g.metal.48xlarge）。

## 动机

PR 解决 #32786：Arm 上 tensor-parallel/data-parallel 跨 NUMA 域推理原先依赖 torch.distributed，同步成本很高；同时缺少开箱即用的线程绑定，导致 OOB（out-of-box）性能差。

## 优化点

在 Arm 上启用 vLLM 自定义 shared memory communicator 取代 torch.distributed，并开启自动线程绑定（auto thread binding）以获得更好 OOB 性能。具体：cmake 在 `ASIMD_FOUND AND NOT APPLE_SILICON_FOUND` 时把 `csrc/cpu/shm.cpp` 加入 CPU 扩展；`cpu_communicator.py` 把 `CpuArchEnum.ARM` 纳入 `init_shm_manager` 的启用条件（原先仅 X86）；`cpu_worker.py` 对 AArch64 按"无 SMT"处理自动绑核；`cpu_types_arm.hpp` 补齐 ASIMD 不支持 non-temporal load 的构造与 INT8Vec64 等向量类型。

## 关键改动

```text
cmake/cpu_extension.cmake: ASIMD 且非 Apple Silicon 时编译 csrc/cpu/shm.cpp
csrc/cpu/shm.cpp（新增）+ torch_bindings.cpp: 自定义共享内存通信器
vllm/distributed/device_communicators/cpu_communicator.py: 通信器启用条件扩展到 ARM
vllm/v1/worker/cpu_worker.py: AArch64 自动线程绑定（无 SMT）
csrc/cpu/cpu_types_arm.hpp: ASIMD non-temporal load 兼容与向量类型
```

```python
         if (
-            (current_platform.get_cpu_architecture() == CpuArchEnum.X86)
+            (
+                current_platform.get_cpu_architecture() == CpuArchEnum.X86
+                or current_platform.get_cpu_architecture() == CpuArchEnum.ARM
+            )
             and hasattr(torch.ops._C, "init_shm_manager")
             and (unique_name.startswith("tp") or unique_name.startswith("pp"))
         ):
```

## 性能声明

基准在 `c8g.metal.48xlarge`，数值为相对基线的百分比；基线 = Single NUMA、tp=1（96 cores、`VLLM_CPU_OMP_THREADS_BIND=0-95`）。正文表格：2 NUMA tp=1（bind 0-191）req/s 134.3% / TTFT 55.4% / TPOT 83.9%；[Current] 2 NUMA tp=2 带绑定 60.2% / 119.3% / 196.4%；[Current] 2 NUMA tp=2 无绑定（OOB, nobind）39.8% / 255.4% / 185.0%；**With PR: 2 NUMA tp=2（191 cores，vLLM 自动绑核）196.3% / 46.7% / 53.6%**。正文总结：带此 PR 跑 tp=2 双 NUMA，**throughput 2x higher、latency 2x lower**（对比单 NUMA tp=1）；**OOB throughput 4x higher、OOB latency 4x lower**（对比当前 tp=2 双 NUMA）。Test Plan 使用 `llama3.2-3b-Instruct`；PR 未注明精度。

## 验证与回归风险

Test Plan：以 `pp=2, tp=1` 与 `pp=1, tp=2` 服务 `llama3.2-3b-Instruct` 并投放 prompts；Test Result：生成文本与 `pp=1, tp=1` 相同。PR 未列专门单元测试。风险面：改动以 Arm/ASIMD 为条件（cmake 条件排除 Apple Silicon），新增 shm.cpp 与 CPU 类型适配；自动绑核对 AArch64 假设无 SMT；通信器切换只作用于 `unique_name` 以 tp/pp 开头的路径。

## 关联

PR 正文标注 `Fixes: #32786`，基准数据亦来自 #32786。素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
