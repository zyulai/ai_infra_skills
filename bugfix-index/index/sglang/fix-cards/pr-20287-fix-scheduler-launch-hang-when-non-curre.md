# PR #20287: fix: scheduler launch hang when non-current rank dies

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20287](https://github.com/sgl-project/sglang/pull/20287) |
| 作者 | alphabetc1 |
| 合并日期 | 2026-03-29 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 17 |
| 分类 | Overlap / Scheduler |
| bug 类型 | hang |

## 一句话结论
launcher 初始化等待只阻塞当前 rank 的 recv,其他 rank 被杀时无限挂起;改为 poll 轮询并检查所有 scheduler 进程存活,死亡立即报错。

## 问题现象
TP 启动时,如果 rank 0 存活而更高编号的 rank 进程被杀死,launcher 会无限期阻塞在 `pipe.recv()` 上等待永远不会到来的响应,没有任何错误信息,整个 launch 挂死。正文给出复现方法:用远低于所有 TP worker 总 `--hicache-size` 的内存上限启动容器(如 `docker run --gpus all --memory=100g --memory-swap=100g ...`),然后 `python -m sglang.launch_server --model-path Qwen/Qwen3-32B-FP8 --enable-hierarchical-cache --mem-fraction-static 0.1 --tp 8 --hicache-size 15`。worker 逐个分配内存时,OS OOM killer 会静默 SIGKILL 后面的一些 worker;若 rank 0 恰好存活,launcher 就一直挂着。

## 根因
正文披露:初始化期间的等待循环只监控它当前正在读取的那个 rank;任何其他 rank 被杀死时,进程会无限阻塞在 `recv()` 上,导致整个 launch hang。

## 修复方案
改 `python/sglang/srt/entrypoints/engine.py` 的 `_wait_for_scheduler_ready`:对每个 pipe reader 用 `poll(timeout=5.0)` 轮询代替阻塞 `recv()`;每次 poll 超时后检查**所有** scheduler 进程(`scheduler_procs[j].is_alive()`),发现任一 rank 死亡就 `join(timeout=10)` 并抛出带 exit code 的 `RuntimeError`,exit code 为 -9(SIGKILL)时提示常见原因是 OS OOM killer、可用 `dmesg -T | grep -i oom` 排查:

```python
if not scheduler_procs[j].is_alive():
    scheduler_procs[j].join(timeout=10)
    raise RuntimeError(
        f"Rank {j} scheduler died during initialization "
        f"(exit code: {scheduler_procs[j].exitcode}). "
        f"If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. "
        ...
    )
```

`status != "ready"` 的检查改为收集完全部 `scheduler_infos` 后统一进行;`EOFError` 分支同样改为抛出带 exit code 与 OOM 提示的 `RuntimeError`。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 launcher 初始化等待逻辑(`_wait_for_scheduler_ready`);等待方式由阻塞 recv 变为 5 秒粒度的 poll + 全量进程存活检查,正常路径下每 rank 至多增加一个 poll 周期的开销;错误消息从泛化报错变为含 exit code 与 OOM 排查指引。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
