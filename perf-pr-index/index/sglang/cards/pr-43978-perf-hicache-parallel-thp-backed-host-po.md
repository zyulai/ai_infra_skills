# PR #43978: [Perf] hicache: Parallel THP-backed host pool allocation (opt-in)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @whybeyoung |
| 合并 | 2026-10-27 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | documentation, hicache, run-ci, bypass-maintenance |
| Diff | diffstat 缺失(素材未统计行数);diff 覆盖 mmap_allocator、environ、文档与单测 4 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/43978 |

## 一句话结论

HiCache host 池改 MAP_PRIVATE+THP 多线程 prefault(默认关),4x B300 节点 306 GB/rank 的分配+钉页从约 185 s 降到 46 s。

## 动机

HiCache host 池由 `alloc_mmap` 以 `MAP_SHARED | MAP_ANONYMOUS` 单线程 `MAP_POPULATE` 预缺页、再 `cudaHostRegister` 钉页。匿名共享映射即 shmem:每次 page fault 都要拿 backing shmem 文件的锁,单线程 prefault 只有约 1.5-2 GB/s,32-64 线程也仅到约 2-4.5 GB/s;且 shmem THP 默认关闭(`shmem_enabled=never`),每页仅 4 KB。host 池很大时这成为引擎启动的显著开销:单节点 4x B300、每 rank 约 306 GB host 池时,分配+钉页占每次启动约 185 s。

## 优化点

新增环境变量 `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS`(默认 0 = 关闭,不设时代码路径不变)。开启后非 hugetlb 路径走 `_mmap_private_parallel_prefaulted`:以 `MAP_PRIVATE | MAP_ANONYMOUS` 映射(普通匿名内存,无 shmem 锁竞争);`madvise(MADV_HUGEPAGE)` 使其在常见 `enabled=madvise` 策略下由 2 MB THP 支持(best effort,4 KB 页仍正确);`madvise(MADV_DONTFORK)` 是 MAP_PRIVATE 可安全钉页的关键——原路径用 MAP_SHARED 正是因为 private 页在 fork 子进程中会 COW、父进程 fork 后的写会把页从已钉页挪走,而 DONTFORK 让映射完全不被继承、无 COW 竞争(与 CUDA driver 对 `cudaHostAlloc` 的处理一致),代价是 fork() 的子进程看不到该缓冲;返回前按 THP 对齐分块、由 n 个线程(实测 32)并行 `memset` 预缺页(ctypes 在 libc 调用期间释放 GIL;实测比每块 `MADV_POPULATE_WRITE` 更快)。任何 OSError 记日志并回退既有 shared prefaulted 映射;显式 hugetlb 路径(`SGLANG_HUGEPAGE_SIZE`)不动。`utils/weight_checker.py` 的 snapshot arena 也走 `alloc_mmap`,因此同样受该 flag 控制。

## 关键改动

```text
python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py: 新增 _mmap_private_parallel_prefaulted(private + MADV_HUGEPAGE/DONTFORK + 并行 memset prefault)与 alloc_mmap 接线、OSError 回退
python/sglang/srt/environ.py: SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS = EnvInt(0)
docs/docs/references/environment_variables.mdx: 新变量文档(含 fork 子进程不可见语义)
test/registered/unit/mem_cache/test_mmap_allocator.py: 常驻/清零、fork SIGSEGV、开关与回退用例
```

## 性能声明

端到端:单 B300 节点,4 rank 并发分配,306 GB host pool/rank,真实 `alloc_mmap` + `_cuda_host_register` 路径,两轮(每 rank):

| | allocation + prefault | cudaHostRegister | total(最慢 rank) |
| --- | --- | --- | --- |
| 默认(MAP_SHARED,单线程 MAP_POPULATE) | 115-172 s | 13-47 s | 185-189 s |
| SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS=32 | 4-27 s | 12-29 s | 46 s |

整体约 4x(每次启动省约 140 s);4 rank 并发时分配受内存带宽/页分配器而非线程数限制。

消融(64 GB prefault,单 B300 节点,256 核,kernel 6.8,THP `enabled=madvise`,`shmem_enabled=never`,分块 memset,两轮):shared + MAP_POPULATE 单线程 1.4-1.8 GB/s;shared 32/64 线程 1.7-4.5/1.9-4.6;private 4KB(MADV_NOHUGEPAGE)32 线程 7.6-8.9;private + THP(100% 2MB 页)32 线程 30-43、64 线程 47-48 GB/s。结论:仅多线程在 shared 映射上最多约 2x 且噪声大,转 private 带来主要扩展性,THP 再叠加约 4x。

## 验证与回归风险

准确性:N/A——纯 host 内存分配,池内容与之前一样精确清零,不触碰模型输出代码。新增 CPU CI 单测:1/4/64 线程下含 ragged 非 THP 对齐大小的映射完全常驻且清零;fork 子进程触碰缓冲得到 SIGSEGV(变异检查:去掉 madvise 该测试失败);flag 开启时 `alloc_mmap` 不走 shared 路径、新路径 OSError 时回退;flag 未设(默认)时绝不走新路径。风险:MADV_DONTFORK 语义变化——fork()(非 spawn)出的子进程无法访问该缓冲,依赖 fork 共享 host 池的部署需注意;THP 依赖系统 `enabled=madvise` 策略,否则退化为 4KB 页仍正确;`weight_checker` snapshot arena 一并受 flag 影响。CI:Base 与 AMD ROCm 10 通过,Extra 失败(素材所示)。

## 关联

同区域:既有 `SGLANG_HUGEPAGE_SIZE` 显式 hugetlb 分配路径(本 PR 不改动,作为并行路径之外的正交选项);HiCache host pool 分配基础设施。其他同分类 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
