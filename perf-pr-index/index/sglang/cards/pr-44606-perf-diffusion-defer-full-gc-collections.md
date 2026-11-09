# PR #44606: [Perf] diffusion: Defer full GC collections while importing the runtime

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rwang5203 |
| 合并 | 2026-10-18 |
| 分类 | Other(major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 显示 4 个文件:process.py、registry.py、worker_bootstrap.py、test_process.py |
| 链接 | https://github.com/sgl-project/sglang/pull/44606 |

## 一句话结论

Diffusion 进程批量导入 runtime 时推迟 CPython full GC:每进程启动少跑 4 次 full collection,省约 0.7s CPU、完整启动延迟约低 1.2%。

## 动机

diffusion scheduler 进程在构建 pipeline 前要导入约一万个模块。导入创建大量长寿命的模块、类与函数,几乎不产生垃圾,但 CPython 每当存活堆增长四分之一就运行一次 full collection,每次都要重新追踪全部受追踪对象——scheduler 进程启动期间要跑 12 次,合计约 1.6 秒 CPU,有无编译字节码都一样。且这些 collection 落在哪里取决于导入顺序而非实际工作量:移动一个 import 就可能在某段启动区间内增减一次整堆 collection,使 import 相关的耗时变化难以评估。

## 优化点

新增 `defer_full_gc_for_imports()` 上下文管理器:进入时把 GC 第三个 threshold 抬到 2^31−1(收集器接受的 C int 上限),使块内不触发任何自动 full collection;young collection 照常运行,短寿命循环垃圾照常回收。退出(包括异常退出)时恢复进入时记录的 thresholds,自动 full collection 恢复;嵌套/并发块在最后一个退出时恢复;块内 fork 的子进程经 `os.register_at_fork` 在 fork 时恢复;collector 的 enabled 状态永不改动。CPython 3.14.0–3.14.4 使用 incremental GC、忽略第三 threshold,该上下文管理器在此版本区间为 no-op。应用于 diffusion 进程批量导入 runtime 的三处:`bootstrap_scheduler_process` 激活阶段(平台初始化到 `gpu_worker` import)、`bootstrap_http_server_process` 激活阶段、registry 的 pipeline/config discovery(pipeline discovery 仅首次进入,早退查找不受影响)。作者还测量并否决了两种更简单的方案:`gc.disable()` 虽省同样 CPU 但停掉 young collection,约 7MB 短寿命循环垃圾存活到块结束(HTTP server 子进程多 8 MiB 匿名内存);`gc.freeze()` 会永久免除 import 对象被回收,改变稳态 serving 内存。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/utils/process.py: 新增 defer_full_gc_for_imports
  上下文管理器(锁 + deferral 计数 + threshold2 抬高/恢复 + at-fork 恢复)
python/sglang/multimodal_gen/runtime/managers/worker_bootstrap.py: scheduler 与 HTTP
  bootstrap 的激活阶段导入包进该块
python/sglang/multimodal_gen/registry.py: pipeline/config discovery 包进该块
python/sglang/multimodal_gen/test/unit/test_process.py: 新增 GC 行为单测
```

关键摘录(核心机制):

```python
with _full_gc_lock:
    if _full_gc_deferrals == 0:
        _gc_thresholds = gc.get_threshold()
        gc.set_threshold(*_gc_thresholds[:2], _NO_FULL_GC)
    _full_gc_deferrals += 1
try:
    yield
finally:
    if os.getpid() == pid:
        with _full_gc_lock:
            _full_gc_deferrals -= 1
            if _full_gc_deferrals == 0:
                gc.set_threshold(*_gc_thresholds)
```

## 性能声明

CPU-only host,按 server 实际启动方式起进程,8 组 counterbalanced 对(Cosmos3 Nano)+ 8 组独立确认(Wan2.1 T2V 1.3B),各配 A/A 对照,仅在两组均 ≥1% 且 A/A 无差异时列收益:scheduler child CPU time −2.2%..−3.0%(95% 区间),wall time −4.5%..−5.8%;HTTP child(--webui)CPU time −2.2%..−2.9%,wall time −4.7%..−5.6%(from source 的 wall time 无 claim,因 A/A 对照读出 −0.53%)。绝对值:scheduler child 少约 0.7s CPU(总约 26–28s),full collection 从 12 次降到 8 次,GC 内耗时少约 0.8s。

真实 server 复核:单张 H200,Wan2.1-T2V-1.3B-Diffusers,FA3/speed mode/无 torch.compile,A/B/B/A 各两次:进程启动→server ready 41.97→41.47s,启动→首个请求完成 42.28→41.78s,即约 0.50s(1.2%)更低完整启动延迟;作者明确说明这是启动收益、不构成通用加速保证,也不建立稳态推理加速。所有进程(含 launcher)匿名内存/峰值 RSS/minor page faults/full collection 次数均不劣于改动前;4 个输出 MP4 逐字节一致。

## 验证与回归风险

`test_process.py` 新增 4 个测试:块内无 full collection 且 threshold 恢复、CPython 3.14.0–3.14.4 incremental GC 不受影响、嵌套/异常退出/GC 禁用状态保持、fork 子进程恢复。该文件在 CPython 3.11.16/3.12.15/3.13.15/3.14.0/3.14.4/3.14.5 通过(3.14.0–3.14.4 上仅跳过 threshold2 相关断言);CPU diffusion 单测与 base 逐例结果一致。回归风险:无输出路径变化,只改启动期 GC 时机;不新增选项,对 srt 无影响;launcher 与 HTTP frontend 进程块内本无 full collection,无性能声明。

## 关联

同区域:#40386(diffusion Hopper fusions)等其他 diffusion runtime 优化;pack 内未列出直接前序/后续 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
