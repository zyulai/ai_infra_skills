# PR #35071: [ROCm][CI] Expose tests to AMD production CI and fix amdsmi heap corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#35071](https://github.com/vllm-project/vllm/pull/35071) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-02-28 |
| labels | rocm, ready, ci/build |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm 测试暴露到 amdproduction CI;修复 amdsmi 每次内存查询反复 init/shutdown 引发的并发堆损坏(加线程锁)。

## 问题现象
ROCm 上的分布式测试间歇性崩溃,报错为:

```
malloc(): unsorted double linked list corrupted
Fatal Python error: Aborted
  File "amdsmi_interface.py", line 1901 in amdsmi_init
```

正文说明其成因链:`tests/utils.py` 的 `_nvml()` 上下文管理器在每次 GPU 内存查询时都调用 `amdsmi_init()`/`amdsmi_shut_down()`;当 `RemoteOpenAIServer` 在 server 启动/关闭期间从多个并发线程查询内存时,这些非线程安全的调用相互竞争,破坏堆内存。该问题在 MI325 上造成 flaky 失败。此外本 PR 同时把多组测试暴露到 `amdproduction` 镜像硬件信号,扩大 AMD 生产 CI 覆盖。

## 根因
正文披露:`_nvml()` 每次进入/退出都执行 `amdsmi_init()`/`amdsmi_shut_down()`,而 amdsmi 的这对调用非线程安全;并发线程同时查询 GPU 内存时产生竞态,导致 malloc 堆结构损坏并 Abort。

## 修复方案
1. `tests/utils.py`(ROCm 分支):模块级创建 `_amdsmi_lock = threading.Lock()`,`_nvml()` 内全程持锁,把 `amdsmi_init()`/`yield`/`amdsmi_shut_down()` 串行化。关键 diff 摘录:

```python
_amdsmi_lock = threading.Lock()

@contextmanager
def _nvml():
    with _amdsmi_lock:
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
```

正文还描述了另一层方案:模块导入时初始化一次 amdsmi 并用 `atexit` 注册关闭,并对 `pynvml` 应用同样模式、不改调用方接口;但 pack 的 diff 中可见的改动仅为上述加锁部分(未显示 init-once/atexit 与 pynvml 对应改动)。

2. `.buildkite/test-amd.yaml`:数十个 test label 的 `mirror_hardwares` 从 `[amdexperimental]` 扩为 `[amdexperimental, amdproduction]`(个别从 `[amdproduction]` 扩为两者、multinode 加入 amdproduction),并为多数步骤新增 `optional: true`。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:测试基建层面——ROCm 上所有经 `_nvml()` 的 GPU 内存查询被锁串行化,并发查询吞吐可能下降但不再竞态;CI 配置层面,AMD 生产流水线将跑更多用例(多数标 optional,不阻塞)。CUDA/pynvml 路径在可见 diff 中未改动。

## 关联
正文未列出 issue/PR 链接(diff 的 CI 配置注释中提及 https://github.com/vllm-project/vllm/pull/31040)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
