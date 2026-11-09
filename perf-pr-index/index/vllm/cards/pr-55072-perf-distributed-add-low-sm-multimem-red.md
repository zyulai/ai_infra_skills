# PR #55072: [Perf][Distributed] Add low-SM multimem reduce-scatter for SM100/SM103

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-25 |
| 分类 | Other (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含约 7 个文件,另有 3 个文件因超长被省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/55072 |

## 一句话结论

为 SM100/SM103 MNNVL 的 TP2/4/8 新增低 SM 数 multimem reduce-scatter 后端,64 MiB 消息较 NCCL 快 7.7%。

## 动机

TP2/TP4/TP8 的 MNNVL reduce-scatter 此前只有两条路径:小消息走现有 MNNVL Lamport 实现(上限 16 MiB),更大消息直接回落 NCCL。16-64 MiB 的中等尺寸消息正好落在 Lamport 不覆盖、NCCL 又非最优的空档,缺少利用 NVLS multicast 的专用实现。该 PR 补齐这条路径,并把所有不支持场景 fail-closed 回退到现有后端。

## 优化点

新核函数用 `multimem.ld_reduce` 指令经 NVLS multicast mapping 直接跨 rank 归约,再用普通 local store 写出本 rank 拥有的输出分片;每 GPU 最多 8 个 CTA、1024 线程、8 路 vector unroll,是刻意压低 SM 占用的 low-SM 设计。调度器按输入总字节数选后端:≤16 MiB 走 Lamport;>16 MiB 且 ≤64 MiB 且 gate 通过走 multimem;>64 MiB 返回 None 回落 NCCL。gate 要求 TP2/4/8、精确 SM100 或 SM103、BF16/FP16/FP32、16 字节分片对齐、group 级 NVLS multicast mapping、非 batch-invariant 执行。64 MiB 对称 staging buffer 在首个符合条件的 reduce-scatter 时惰性分配;所有 rank 先归约各自分配结果再进入 symmetric-memory rendezvous,任一 rank 失败则整组回退,不会搁浅 peers。

## 关键改动

```text
csrc/custom_all_gather_reduce_scatter.cuh: multimem.ld_reduce/multimem.st PTX 封装与 reduce-scatter 主核,ROCm 下整体编译排除
csrc/libtorch_stable/custom_all_gather_reduce_scatter.cu: host 入口,barrier + 主核 + barrier 三段发射,校验 dtype/对齐/block 上限
vllm/distributed/device_communicators/custom_all_reduce.py: 后端选择策略 _select_reduce_scatter_backend、惰性 staging buffer 初始化、_all_ranks_true 集体协商
benchmarks/kernels/benchmark_multimem_reduce_scatter.py: 多 rank 可复用基准,计时前对 FP32 分布式参考校验每个 workspace
```

## 性能声明

8x B300 生产路径计时(含 staging 拷贝与必要 barrier,取各 rank 最大 CUDA event 时间的中位数):64 MiB hot 137.38 us vs NCCL 148.83 us(7.7% faster);64 MiB cold L2 141.18 vs 150.27 us(6.0% faster);76 MiB hot 161.79 vs 162.21 us(0.3% faster);96 MiB hot 198.53 vs 180.67 us(9.9% slower)。64 MiB 上限刻意低于实测约 76 MiB 的 hot-cache 交叉点。FP16/FP32 在 64 MiB 下 multimem 路径均测得 137.25 us。测试硬件为 8x B300,BF16/FP16/FP32,消息 64-96 MiB,hot 与 cold L2 两种状态。

## 验证与回归风险

新增 pytest 门控测试(mnnvl_multimem_reduce_scatter_platform_gate、mnnvl_reduce_scatter_backend_gate、cross_node_mnnvl_gate,39 passed/10 deselected);TP2/TP4/TP8 集成测试 3 passed,验证惰性 staging 分配、1 MiB Lamport、20 MiB multimem 与 >64 MiB NCCL 回退;BF16 cutoff sweep 及 FP16/FP32 64 MiB 正确性均对照 FP32 分布式参考;CUDA graph capture/replay 在 TP2/4/8 下逐 workspace 验证后才计时;SM100 代码审计为 62 registers/thread、128 字节 stack、零 shared/local memory。风险集中在 collective 传输选择层,不改模型计算;最多新增 64 MiB 对称内存占用,不支持拓扑/dtype/对齐均 fail-closed。

## 关联

同区域相关 PR:#50182(调整 symmetric-memory multicast 选择)、#44080(调整 sequence-parallel fusion 限制)、#53852(FlashInfer all-reduce cutoff override)。作者声明经 Open PR/issue 搜索未发现重复实现。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
