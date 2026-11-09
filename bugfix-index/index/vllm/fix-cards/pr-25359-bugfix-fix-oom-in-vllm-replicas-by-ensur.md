# PR #25359: [BugFix] Fix OOM in vLLM replicas by ensuring consistent NCCL memory accounting

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#25359](https://github.com/vllm-project/vllm/pull/25359) |
| 作者 | kouroshHakha |
| 合并日期 | 2025-09-23 |
| labels | ready, ci/build, v1 |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Parallelism / Communication |
| bug 类型 | resource |

## 一句话结论
内存快照先于 NCCL 初始化,部分副本 KV cache 计算偏大导致间歇 OOM;把分布式环境初始化挪到快照之前。

## 问题现象
用 ray distributed backend 部署多个 vLLM 副本时(作者不确定 ray 是否为成因之一),间歇性出现 OOM:配置完全相同的情况下,部分副本因 KV cache 内存不足而失败,其他副本正常。副本有时看到 78.59 GiB 空闲内存(NCCL 初始化前),有时看到 74.07 GiB(初始化后),KV cache 计算随之失败。

## 根因
正文披露(作者注明这是 viable hypothesis):GPU worker 的 `init_device()` 中存在竞态。1) 线程问题:使用 Ray Compiled DAG 时模型执行在后台线程,而内存快照在主线程拍取;2) NCCL 初始化时序:TP 下约 4.5 GiB 的 NCCL buffer 有时在内存快照之后才分配,导致内存测量不一致,NCCL 分配在 forward 期间表现为 non-torch 内存增长;3) 上述时序差异造成副本间内存核算不一致。

## 修复方案
重排 `vllm/v1/worker/gpu_worker.py` 的 `init_device()` 初始化顺序:把 `init_worker_distributed_environment()` 与 `set_random_seed()` 从函数尾部挪到设备检查之后、`gc.collect()`/`torch.cuda.empty_cache()` 及内存快照之前,保证 NCCL buffer 分配完成后再测量:

```diff
+            # Initialize the distributed environment BEFORE taking
+            # memory snapshot
+            init_worker_distributed_environment(self.vllm_config, self.rank,
+                                                self.distributed_init_method,
+                                                self.local_rank,
+                                                current_platform.dist_backend)
+            set_random_seed(self.model_config.seed)
+            # Now take memory snapshot after NCCL is initialized
             gc.collect()
             torch.cuda.empty_cache()
```

(原函数尾部的两处调用被删除。)另新增 `tests/v1/worker/test_worker_memory_snapshot.py`:多进程下用队列记录操作次序,断言 `init_distributed < nccl_all_reduce < memory_snapshot`;`.buildkite/test-pipeline.yaml` 把该测试加入两条流水线。作者说明 NCCL 强制初始化(对 TP 做一次 dummy `all_reduce`)本就存在于 pynccl communicator init 中。实测 Qwen/Qwen2.5-0.5B-Instruct TP=2(4xL4):Available KV cache 17.77→17.85 GiB,non-torch forward increase 0.13→0.05 GiB,profiling 耗时 22.90→7.31 s。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`init_device()` 的初始化顺序——分布式环境初始化现发生在 `torch.cuda.empty_cache()`/内存快照之前,任何依赖「分布式初始化晚于设备/内存操作」的假设都会改变(由 diff 推断);NCCL buffer 计入快照后,可用 KV cache 计算普遍略变(本例 +0.08 GiB);新增测试依赖 ≥2 GPU,单卡 CI 跳过。

## 关联
正文未列出(正文内引用了 vllm/distributed/device_communicators/pynccl.py 的 pynccl communicator 代码链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
