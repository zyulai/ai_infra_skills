# PR #36955: [Bugfix] Fix unclean shutdown crash with AllReduce Fusion workspace

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36955](https://github.com/vllm-project/vllm/pull/36955) |
| 作者 | siewcapital |
| 合并日期 | 2026-03-17 |
| labels | bug, nvidia, ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Parallelism / Communication |
| bug 类型 | crash |

## 一句话结论
ctrl-c 时 FlashInfer AllReduce workspace 在解释器关闭期 __del__ 崩溃;改用 atexit 在关闭前释放。

## 问题现象
B200 上启用 allreduce fusion 后按 ctrl-c,FlashInfer `AllReduceFusionWorkspace.__del__` 崩溃:
```
ImportError: sys.meta_path is None, Python is likely shutting down
```

## 根因
正文披露:全局 workspace 对象(`_fi_ar_workspace`、`_fi_ar_quant_workspace`)依赖解释器关闭期间的垃圾回收 `__del__` 来清理,但彼时 Python 的模块/导入系统已被拆除,于是抛出 ImportError。

## 修复方案
`vllm/distributed/device_communicators/flashinfer_all_reduce.py`:注册 `atexit.register(destroy_fi_ar_workspace)`,在 Python 开始解释器关闭之前释放全局 workspace 引用,保证清理执行时导入系统仍可用;`destroy_fi_ar_workspace` 加 `threading.Lock` 保护:
```python
_fi_ar_workspace_lock = threading.Lock()

def destroy_fi_ar_workspace():
    global _fi_ar_workspace
    global _fi_ar_quant_workspace
    with _fi_ar_workspace_lock:
        ...
        if _fi_ar_workspace is not None:
            _fi_ar_workspace.destroy()
            _fi_ar_workspace = None

atexit.register(destroy_fi_ar_workspace)
```
与 `vllm/distributed/device_communicators/pynccl_allocator.py` 已有的同类 shutdown 处理模式保持一致。作者自检:flashinfer 未安装时清理函数对 None 全局不做任何操作;`destroy_fi_ar_workspace` 幂等,可安全多次调用。

## 影响与回归风险
- 回归面:仅进程关闭/清理路径;不改变 allreduce fusion 的运行时行为;atexit 注册使正常退出也会先走显式清理。

## 关联
Fixes #35686

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
