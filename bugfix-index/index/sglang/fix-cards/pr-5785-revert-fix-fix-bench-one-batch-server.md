# PR #5785: Revert "[fix] fix bench_one_batch_server"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5785](https://github.com/sgl-project/sglang/pull/5785) |
| 作者 | merrymercy |
| 合并日期 | 2025-04-27 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
回滚 #5607:其在模块顶层强制设置 multiprocessing spawn 全局启动方式,破坏部分 CI 测试,删除该行恢复默认行为。

## 问题现象
部分 CI 测试被 #5607(fix bench_one_batch_server)破坏(正文:"It sets some global behavior. It breaks some CI tests")。正文未列出具体失败的测试名与报错。

## 根因
正文披露:#5607 设置了某种全局行为(set some global behavior),因而破坏 CI 测试。从 diff 看,该全局行为是 `python/sglang/bench_one_batch_server.py` 模块顶层的:

```python
multiprocessing.set_start_method("spawn", force=True)
```

`force=True` 会在导入该模块时强制改写整个进程的 multiprocessing 启动方式(由 diff 推断具体机制)——任何导入 bench_one_batch_server 的进程(包括 CI 中共用进程的其他测试)都会被波及,与 #5607 只想影响 bench 自身的意图不符。

## 修复方案
删除 `python/sglang/bench_one_batch_server.py` 顶部的一行:

```diff
 from sglang.srt.server_args import ServerArgs
 from sglang.srt.utils import kill_process_tree

-multiprocessing.set_start_method("spawn", force=True)

```

multiprocessing 启动方式回到 Python 默认,bench 脚本本身的其他逻辑不动。

## 影响与回归风险
- 影响窗口:#5607 合并至本回滚之间。
- 回归面:仅 `bench_one_batch_server.py` 的导入副作用;运行时与服务端代码不受影响。#5607 想通过 spawn 达成的 bench 行为(正文未说明其动机细节)不再由全局设置保证,若 bench 子进程行为有差异需在其内部自行处理。

## 关联
- [#5607](https://github.com/sgl-project/sglang/pull/5607)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
