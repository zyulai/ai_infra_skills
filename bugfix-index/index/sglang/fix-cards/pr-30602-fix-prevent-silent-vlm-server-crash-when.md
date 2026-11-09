# PR #30602: [Fix] Prevent silent VLM server crash when /dev/shm is exhausted during multimodal feature transport

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30602](https://github.com/sgl-project/sglang/pull/30602) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-07-09 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

多模态请求突发打满 /dev/shm 时进程被 SIGBUS 杀死;预预留页并回退 inline 传输,改为可捕获的降级。

## 问题现象

在突发的一批多模态请求下,`ShmPointerMMData` 可能在拷贝过程中耗尽 `/dev/shm`,直接以 SIGBUS 杀死服务进程。原因是共享内存段只被 `ftruncate`,tmpfs 页面是写入时惰性分配的,写到一半空间不足就会触发 SIGBUS,进程无法捕获。

## 根因

正文披露:`ShmPointerMMData` 分配共享内存后,`SharedMemory` 只 ftruncate 段大小,tmpfs 页在真正写入时才惰性分配;`/dev/shm` 在拷贝中途被填满时内核发 SIGBUS 杀进程,异常无法在 Python 层捕获,也没有降级路径。

## 修复方案

`python/sglang/srt/managers/mm_utils.py` 三处改动:

1. `ShmPointerMMData.__init__` 中,在 `copy_` 之前于 Linux 上调用 `os.posix_fallocate(shm._fd, 0, nbytes)` 预预留页,把"写一半才失败"变成分配时即可捕获的 `OSError`(ENOSPC);
2. 新增 `_wrap_shm_or_inline`:包装失败(`OSError`)时告警一次(提示增大 /dev/shm)并回退为 inline(pickled)传输,`_wrap_tensor_or_list` 的单张量与逐元素路径均改用它;
3. 顺带修 `hash_feature`:列表可能混含 `ShmPointerMMData` 与裸张量(因 shm 分配失败按元素回退 inline),由只检查首元素改为 `any(...)`,并对每元素解包后再哈希。

关键 diff 摘录:

```diff
+            if sys.platform == "linux":
+                # ... if /dev/shm fills up mid-copy
+                # the process is killed with SIGBUS. Reserving the pages up
+                # front turns exhaustion into a catchable OSError (ENOSPC).
+                os.posix_fallocate(shm._fd, 0, nbytes)
```

验证:在本 PR head(`38f5b6d`,1-gpu-h100 runner)对 `test_bench_serving_1gpu_part2.py` 压测 10 次,10/10 通过(附 10 个 CI run 链接)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:CPU 张量的多模态特征传输路径。shm 分配成功时行为不变(posix_fallocate 只是提前预留页);`/dev/shm` 耗尽时从进程死亡变为 inline 回退(吞吐下降但不崩溃);`hash_feature` 对混合列表的哈希结果与纯包装列表一致。仅 Linux 走 fallocate 分支。

## 关联

正文未列出 issue/PR 链接(附 10 个验证用 CI run 链接)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
