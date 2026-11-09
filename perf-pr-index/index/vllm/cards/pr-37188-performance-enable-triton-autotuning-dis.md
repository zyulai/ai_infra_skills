# PR #37188: [Performance] Enable Triton autotuning disk cache by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @arpera |
| 合并 | 2026-03-19 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段实际含 vllm/env_override.py 一处改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/37188 |

## 一句话结论

默认开启 TRITON_CACHE_AUTOTUNING 使重启复用 autotune 结果，8×B200 / Qwen3.5-397B-FP8 第 2 次 benchmark 由 121.49s 降到 39.45s（约 3×）。

## 动机

Triton 的 `@triton.autotune` 装饰器因 `TRITON_CACHE_AUTOTUNING` 默认关闭，会在每次进程重启时重新跑 kernel autotuning。正文指出这给 vLLM serving 每次 server 启动后的首个推理请求带来显著延迟，因此值得把 autotuning 结果持久化并跨重启复用。

## 优化点

在 `vllm/env_override.py` 中通过 `os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")` 把 autotuning 结果写入 `TRITON_CACHE_DIR` 并跨重启复用。使用 `setdefault` 保证用户仍可显式设置 `TRITON_CACHE_AUTOTUNING=0` 退出该行为。改动本身只是一行环境变量默认值，机制层面是把 Triton 的 autotuning disk cache 默认打开，从而跳过重启后的重复 autotune。正文对比：改动前第 2 次启动仍打印 autotuning 日志（Triton 忽略 disk cache、从零重跑）；改动后第 2 次启动无 autotuning 日志，直接从磁盘加载缓存结果。

## 关键改动

```text
vllm/env_override.py: 在 torch._inductor 配置段之后新增 TRITON_CACHE_AUTOTUNING 默认开启的环境变量
```

```diff
+# Enable Triton autotuning result caching to disk by default.
+# Without this, Triton re-runs autotuning on every process restart,
+# adding significant latency to the first inference request.
+# This writes autotuning results to TRITON_CACHE_DIR.
+# It can still be overridden by setting TRITON_CACHE_AUTOTUNING=0
+# in the environment.
+os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
```

## 性能声明

测试条件（Test result 段）：8× NVIDIA B200，`Qwen/Qwen3.5-397B-A17B-FP8`，dp=8、expert parallelism，128 prompts，random input 8000 tokens、output 1 token、concurrency 8、`temperature 0.0`、`--ignore-eos`；对比「2nd run before change」与「2nd run after change」两次重启后的 `vllm bench serve` 结果。Benchmark duration (s)：121.49 → **39.45**；Request throughput (req/s)：1.05 → **3.24**；Total token throughput (tok/s)：8,430.07 → **25,961.99**；Mean TTFT (ms)：7,555.32 → **2,449.88**；Median TTFT (ms)：1,001.79 → **985.23**；P99 TTFT (ms)：90,318.72 → **20,389.16**；Successful requests 128/128、Failed 0/0。正文结论："By caching Triton kernel autotuning results to disk, the 2nd benchmark run achieved a **3x speedup**"（perf claims 段标注 3x speedup）。

## 验证与回归风险

验证方式：PR 以相同 server/bench 命令重启前后两次运行的日志对比（设 `TRITON_PRINT_AUTOTUNING=1` 观察），改动前第 2 次仍出现 autotuning 日志、改动后第 2 次无 autotuning 日志，并给出上表 benchmark 数据。可能影响的路径：所有依赖 Triton autotune 的 kernel（日志中点名 `chunk_local_cumsum_scalar_kernel`）；首次冷启动仍会执行 autotune；`TRITON_CACHE_DIR` 磁盘缓存的持久化行为发生变化。PR 未注明更多回归风险。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
