# PR #4477: fix custom allreduce performance/accuracy problem

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#4477](https://github.com/sgl-project/sglang/pull/4477) |
| 作者 | yizhang2077 |
| 合并日期 | 2025-03-16 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Parallelism / Communication |
| bug 类型 | correctness |

## 一句话结论
custom allreduce 启动配置与 barrier 位置不当导致精度/性能问题;改回 vllm 风格的 threads/blocks 计算与同步时机。

## 问题现象
正文未描述具体现象,仅说明修复 custom allreduce 的 accuracy/performance problem(见 #4441),并以 Llama-3.1-8B-Instruct TP8 的 benchmark 数据对比修复后与 vllm allreduce:custom allreduce 吞吐 101.65 req/s / 输出 19211.30 tok/s,vllm allreduce 99.23 req/s / 18753.56 tok/s;gsm8k 0.788(Invalid 0.001)vs 0.794(Invalid 0.000);mmlu 平均 0.683 vs 0.684。

## 根因
正文未披露根因,只列了两个修改方向。从 diff 描述改动所暗示的机制(由 diff 推断):`kernelLaunchConfig` 原先用 while 循环自增 `blocks_per_grid` 再整除调参,可能推出劣化的 launch 配置(还带 iter_factor 修正 hack);`block_barrier` 中无条件 `__syncthreads()`,而 oneShot kernel 结尾缺少 barrier 同步,与 vllm 的做法不一致。

## 修复方案
`sgl-kernel/csrc/allreduce/trt_reduce_internal.cu`:
1. launch 配置改回 vllm 风格:`threads_per_block = min(DEFAULT_BLOCK_SIZE, total_threads)`,`blocks_per_grid = min(MAX_ALL_REDUCE_BLOCKS, divUp(...))`,删除 while 循环与 iter_factor 修正;
2. `block_barrier` 仅在 `start || need_fence` 时 `__syncthreads()`,`oneShotAllReduceKernel` 尾部补 `block_barrier<false>(...)`。
`trt_reduce_internal.cuh`:`MAX_ALL_REDUCE_BLOCKS` 36 → 32。

```python
-      while (total_threads % blocks_per_grid != 0 || total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE) {
-        blocks_per_grid += 1;
-      }
-      threads_per_block = total_threads / blocks_per_grid;
+      threads_per_block = std::min(DEFAULT_BLOCK_SIZE, total_threads);
+      blocks_per_grid = std::min(static_cast<int>(MAX_ALL_REDUCE_BLOCKS), divUp(total_threads, threads_per_block));
```

## 影响与回归风险
- 影响窗口:正文未说明(精度问题由 #4441 引入)
- 回归面:sgl-kernel 的 one-shot custom allreduce kernel 的 launch 配置与同步语义,影响所有走该算子的 TP 通信路径;与 vllm 对齐后行为应与其一致。

## 关联
- https://github.com/sgl-project/sglang/pull/4441

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
