# PR #31604: [Benchmark] Fix OOM during MoE kernel tuning for large models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#31604](https://github.com/vllm-project/vllm/pull/31604) |
| 作者 | massif-01 |
| 合并日期 | 2026-01-02 |
| labels | performance, ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | resource |

## 一句话结论
`benchmark_moe.py --tune` 调优大 MoE 模型时 Triton JIT 缓存累积致内存从 ~40GB 涨到 100GB+ 触发 Ray OOM;新增每 50 个配置周期性清缓存修复。

## 问题现象
对专家数多的大 MoE 模型(如 128 experts)运行 `benchmarks/kernels/benchmark_moe.py --tune` 时 OOM。触发机制:Triton JIT 为每个配置编译并缓存 kernel,每个 batch_size 有 1920 个配置,内存从约 40GB 增长到 100GB 以上,最终触发 Ray 的 OOM killer。实测 GPT-OSS-120B(128 experts, topk=4)、batch-size 1-64 时,原行为在 batch_size=2 即 OOM。

## 根因
PR 正文披露:OOM 源于调优过程中 Triton JIT 编译缓存的持续累积——每个被搜索的配置都会编译并缓存一份 kernel,配置总数大(每 batch_size 1920 个)时缓存占用随时间线性增长,直至触发 Ray OOM killer;正文未描述更底层的分配细节。

## 修复方案
在 `benchmarks/kernels/benchmark_moe.py` 中增加周期性缓存清理:

1. 新增 `clear_triton_cache()`:依次执行 `gc.collect()`、`torch.cuda.empty_cache()`、清理 Triton runtime cache(`triton.runtime.cache.clear()`,兼容无该 API 的版本,ImportError/AttributeError 静默跳过),最后再 `gc.collect()`。
2. `tune()` 循环改为 `enumerate`,每处理 `TRITON_CACHE_CLEAR_INTERVAL`(默认 50,可用环境变量 `VLLM_MOE_TUNE_CACHE_CLEAR_INTERVAL` 配置,设 0 禁用)个配置调用一次清理:

```diff
+                if (
+                    TRITON_CACHE_CLEAR_INTERVAL > 0
+                    and idx > 0
+                    and idx % TRITON_CACHE_CLEAR_INTERVAL == 0
+                ):
+                    clear_triton_cache()
+
+        # Final cleanup after tuning completes
+        clear_triton_cache()
```

3. 每个 batch_size 调优结束后再做一次最终清理。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:仅影响 benchmark 脚本 `benchmarks/kernels/benchmark_moe.py` 的调优流程,不影响 vLLM 推理运行时;清理间隔可通过环境变量调整或禁用。验证:GPT-OSS-120B(128 experts, topk=4)batch-size 1-64 从 batch_size=2 OOM 变为全部完成;未报告其他模型的调优验证。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
