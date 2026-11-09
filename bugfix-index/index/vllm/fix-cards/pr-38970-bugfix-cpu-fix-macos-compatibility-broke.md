# PR #38970: [Bugfix][CPU] Fix macOS compatibility broken by #36487

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#38970](https://github.com/vllm-project/vllm/pull/38970) |
| 作者 | 2imi9 |
| 合并日期 | 2026-04-04 |
| labels | bug, cpu, ready, verified |
| 热度 | 👍 2 · 💬 1 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#36487 引入对 Linux 专属 API(sched_getaffinity/lscpu)的无条件调用,令 macOS CPU 支持损坏;补平台判断与回退逻辑修复。

## 问题现象
macOS 上 vLLM 的 CPU 支持被破坏,`OMPProcessManager` 无法初始化:PR #36487("Replace OMP initialization")引入了两处对 Linux 专属能力的无条件调用——`os.sched_getaffinity`(macOS 上不存在该属性)和 `lscpu`(Linux-only 命令)。该问题曾由 @hmellor 在 #36487 的 review 评论中指出。修复验证:macOS(Darwin)上 `OMPProcessManager` 可成功初始化、回退路径返回正确 CPU 集合、拓扑 JSON 合法。

## 根因
正文披露:#36487 在 OMP 初始化路径中不加平台判断地调用了 macOS 不存在的 `os.sched_getaffinity` 与 Linux-only 的 `lscpu` 命令,导致非 Linux 平台初始化即失败。

## 修复方案
- `vllm/utils/ompmultiprocessing.py`:
  - 新增 `_get_default_affinity()`:存在 `sched_getaffinity` 时照常使用,否则回退 `os.cpu_count()`;
  - 新增 `_get_cpu_topology_json()`:Linux 上仍执行 `lscpu -Je`,非 Linux(如 macOS)合成"每个逻辑 CPU 独立为一个 core、同属 NUMA node 0"的简单拓扑,供 OMP place-list 构建使用;
  - `enumerate_resources()` 与 `OMPProcessManager.__init__()` 改用上述 helper。
- `vllm/platforms/cpu.py`:`CpuPlatform.get_global_cpu_mask()` 加 `hasattr(os, "sched_getaffinity")` 守卫:
```diff
-            cls.global_cpu_mask = os.sched_getaffinity(0)
+            if hasattr(os, "sched_getaffinity"):
+                cls.global_cpu_mask = os.sched_getaffinity(0)
+            else:
+                # macOS does not support sched_getaffinity
+                cpu_count = os.cpu_count() or 1
+                cls.global_cpu_mask = set(range(cpu_count))
```

## 影响与回归风险
- 影响窗口:#36487 合并之后至本 PR 之前,macOS 上的 CPU 构建受影响。
- 回归面:Linux 路径行为完全不变(`sched_getaffinity` 存在、`lscpu` 照常调用);非 Linux 平台获得回退拓扑(每逻辑 CPU 独立 core、单 NUMA 节点),对依赖真实 NUMA 拓扑的 OMP 资源枚举在非 Linux 上只是近似;`get_global_cpu_mask`/`enumerate_resources`/`OMPProcessManager.__init__` 三条路径均被触及。

## 关联
- PR #36487(引入回归的改动,review 中由 @hmellor 指出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
